# utils/command_logger.py

import subprocess
import logging
import time
import re
from functools import wraps
from threading import Thread
from queue import Queue, Empty

class CommandLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.progress_patterns = [
            r"\d+%|\d+/\d+",
            r"$$.*>.*$$",
            r"\d+\.\d+it/s",
            r"ETA: \d+:\d+",
            r"\d+h\d+m\d+s"
        ]

    def log_command(self, func):
        """装饰器：记录函数调用的命令执行"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.debug(
                f"[CMD WRAPPER] 函数 {func.__name__} 调用\n"
                f"参数: {args}\n"
                f"关键字参数: {kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                self.logger.debug(f"[CMD WRAPPER] 函数 {func.__name__} 执行成功")
                return result
            except subprocess.CalledProcessError as e:
                self.logger.error(
                    f"[CMD WRAPPER] 命令执行失败\n"
                    f"命令: {e.cmd}\n"
                    f"返回码: {e.returncode}\n"
                    f"输出: {e.output}"
                )
                raise
            except Exception as e:
                self.logger.error(
                    f"[CMD WRAPPER] 函数 {func.__name__} 发生未处理异常\n"
                    f"异常类型: {type(e).__name__}\n"
                    f"异常信息: {str(e)}"
                )
                raise
        return wrapper

    def _is_progress_bar(self, line: str) -> bool:
        """智能识别进度条输出"""
        try:
            return any(re.search(p, line) for p in self.progress_patterns)
        except Exception as e:
            self.logger.debug(f"进度条识别异常: {str(e)}")
            return False

    def _is_real_error(self, line: str) -> bool:
        """识别真正的错误信息"""
        error_keywords = ["error", "fail", "exception", "traceback"]
        return any(kw in line.lower() for kw in error_keywords)

    def capture_subprocess(self, command, shell=False, timeout=None, **kwargs):
        def enqueue_output(out, queue):
            """安全读取输出到队列"""
            try:
                while True:
                    line = out.readline()
                    if not line:
                        break
                    queue.put(line.rstrip())
            except (ValueError, OSError) as e:
                self.logger.debug(f"输出读取终止: {str(e)}")
            finally:
                try:
                    out.close()
                except Exception as e:
                    self.logger.debug(f"关闭管道异常: {str(e)}")

        # 记录命令开始
        self.logger.info(f"[CMD START] 执行命令: {' '.join(command)}")

        # 启动子进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            bufsize=1,
            universal_newlines=True,
            **kwargs
        )

        # 创建队列和读取线程
        q_stdout, q_stderr = Queue(), Queue()
        Thread(target=enqueue_output, args=(process.stdout, q_stdout), daemon=True).start()
        Thread(target=enqueue_output, args=(process.stderr, q_stderr), daemon=True).start()

        # 收集输出的缓冲区
        stdout_buf, stderr_buf = [], []
        start_time = time.time()

        try:
            while True:
                # 处理标准输出
                while not q_stdout.empty():
                    line = q_stdout.get_nowait()
                    self.logger.info(f"[CMD STDOUT] {line}")
                    stdout_buf.append(line)

                # 智能处理错误输出
                while not q_stderr.empty():
                    line = q_stderr.get_nowait()
                    
                    if self._is_progress_bar(line):
                        self.logger.info(f"[CMD PROGRESS] {line}")
                    elif self._is_real_error(line):
                        self.logger.error(f"[CMD STDERR] {line}")
                    else:
                        self.logger.info(f"[CMD OUTPUT] {line}")
                    
                    stderr_buf.append(line)

                # 检查超时
                if timeout and (time.time() - start_time) > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)

                # 检查进程状态
                retcode = process.poll()
                if retcode is not None:
                    break

                time.sleep(0.1)

        except subprocess.TimeoutExpired:
            process.kill()
            self.logger.error(
                f"[CMD TIMEOUT] 命令执行超时 ({timeout}秒)\n"
                f"已运行时间: {time.time() - start_time:.2f}秒"
            )
            raise
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception as e:
                    self.logger.debug(f"终止进程异常: {str(e)}")

        # 构建返回对象
        result = subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout='\n'.join(stdout_buf),
            stderr='\n'.join(stderr_buf)
        )

        # 最终状态记录
        if process.returncode == 0:
            self.logger.info(
                f"[CMD FINISH] 命令成功完成\n"
                f"返回码: {process.returncode}\n"
                f"输出行数: {len(stdout_buf)}"
            )
        else:
            error_lines = '\n'.join(stderr_buf[-3:])  # 关键修复点
            error_msg = (
                f"[CMD FAILED] 命令异常结束\n"
                f"返回码: {process.returncode}\n"
                f"最后3条错误输出:\n"
                f"{error_lines}"
            )
            self.logger.error(error_msg)
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output='\n'.join(stdout_buf),
                stderr='\n'.join(stderr_buf)
            )

        return result