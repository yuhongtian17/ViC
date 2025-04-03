# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import os
import glob
import tempfile

class LoggingConfig:
    def __init__(self, 
                 log_file='app.log', 
                 max_size=100 * 1024 * 1024,  # 100MB
                 backup_count=5,
                 clean_existing=False):
        """
        初始化日志配置
        :param log_file: 主日志文件路径
        :param max_size: 单个日志文件最大字节数
        :param backup_count: 保留的备份文件数量
        :param clean_existing: 是否清理历史日志文件
        """
        self.log_file = os.path.abspath(log_file)
        self.max_size = max_size
        self.backup_count = backup_count
        self.clean_existing = clean_existing

    def setup_logging(self):
        """配置完整的日志系统"""
        # 先处理目录和清理
        self._ensure_log_dir()
        
        if self.clean_existing:
            self._clean_existing_logs()

        # 配置日志处理器
        formatter = self._create_formatter()
        self._configure_root_logger(formatter)

        # 设置第三方库日志级别
        self._set_thirdparty_log_level()

    def _clean_existing_logs(self):
        """清理所有关联的日志文件"""
        try:
            # 删除主日志文件
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                logging.debug(f"已删除主日志文件: {self.log_file}")

            # 使用通配符匹配所有备份文件
            pattern = f"{self.log_file}.*"
            for f in glob.glob(pattern):
                os.remove(f)
                logging.debug(f"已删除历史备份文件: {f}")

        except Exception as e:
            logging.error(f"清理日志文件失败: {str(e)}")
            raise

    def _ensure_log_dir(self):
        """确保日志目录存在且可写"""
        log_dir = os.path.dirname(self.log_file)
        try:
            os.makedirs(log_dir, exist_ok=True)
            if not os.access(log_dir, os.W_OK):
                raise PermissionError(f"目录不可写: {log_dir}")
        except Exception as e:
            # 异常时回退到临时目录
            new_path = os.path.join(
                tempfile.gettempdir(), 
                f"app_{os.getpid()}.log"  # 添加进程ID防止冲突
            )
            logging.error(f"日志目录异常，已回退至: {new_path}")
            self.log_file = new_path

    def _configure_root_logger(self, formatter):
        """配置根日志记录器"""
        root_logger = logging.getLogger()
        
        # 清理现有处理器
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        # 创建新处理器
        handlers = [
            self._create_file_handler(formatter),
            self._create_console_handler(formatter)
        ]

        # 配置日志记录器
        root_logger.setLevel(logging.DEBUG)
        for handler in handlers:
            if handler:
                root_logger.addHandler(handler)

    def _create_file_handler(self, formatter):
        """创建文件日志处理器"""
        try:
            handler = RotatingFileHandler(
                filename=self.log_file,
                mode='w',          # 强制覆盖模式
                maxBytes=self.max_size,
                backupCount=self.backup_count,
                encoding='utf-8',
                delay=False        # 立即创建文件
            )
            handler.setFormatter(formatter)
            return handler
        except Exception as e:
            logging.error(f"文件处理器创建失败: {str(e)}")
            return None

    def _create_console_handler(self, formatter):
        """创建控制台处理器"""
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
        return handler

    @staticmethod
    def _create_formatter():
        """创建标准格式器"""
        return logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    @staticmethod
    def _set_thirdparty_log_level():
        """设置第三方库的日志级别"""
        for lib in ['urllib3', 'requests', 'PIL', 'matplotlib']:
            logging.getLogger(lib).setLevel(logging.WARNING)

if __name__ == '__main__':
    # 示例用法
    config = LoggingConfig(
        log_file="./log/app.log",
        clean_existing=True
    )
    config.setup_logging()
    
    logging.info("程序启动日志")
    logging.warning("测试警告信息")