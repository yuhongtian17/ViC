from flask import Flask, render_template, request, send_file, jsonify, send_from_directory

import os
import sys
import logging
import threading
import traceback
import time
from datetime import datetime
from glob import glob

from utils.logging_config import LoggingConfig
from utils.command_logger import CommandLogger


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #


UPLOAD_FOLDER = './uploads/'
OUTPUT_FOLDER = './output/'
CONFIG_FOLDER = './mmdetection-3.3.0/configs/_test_/'
BASE_CONFIG_PATH = './utils/old_config.py'
MODEL_WEIGHTS = './pth/epoch_12.pth'

# 创建必要目录
required_dirs = [
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    CONFIG_FOLDER,
    os.path.join(OUTPUT_FOLDER, 'pkl'),
    os.path.join(OUTPUT_FOLDER, 'xlsx'),
]

for d in required_dirs:
    os.makedirs(d, exist_ok=True)
    logging.info(f"Directory created: {d}")


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #


log_config = LoggingConfig(log_file=os.path.join(os.getcwd(), './log/app.log'),clean_existing=True)
log_config.setup_logging()  # 必须调用此方法激活配置
cmd_logger = CommandLogger()

app = Flask(__name__, template_folder='./templates/',static_folder='./static')  # 添加这行

# 配置应用参数
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 全局异常处理
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logging.critical("Uncaught global exception",
                    exc_info=(exc_type, exc_value, exc_traceback),
                    stack_info=True)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

# Flask错误处理
@app.errorhandler(Exception)
def handle_flask_exception(e):
    logging.error("Request handling exception",
                 exc_info=True,
                 extra={
                     'url': request.url,
                     'method': request.method,
                     'ip': request.remote_addr
                 })
    return render_template('error.html', error=str(e)), 500


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #


@app.route('/', methods=['GET', 'POST'])
def index():
    """主路由处理（修正装饰器位置）"""

    @cmd_logger.log_command  # 正确：装饰器应用在内部函数
    def handle_root_file(file):
        """处理ROOT文件上传（添加命令日志装饰器）"""
        try:

            # 检测上传文件是否为ROOT文件格式
            filename = file.filename
            if filename[-5:] != '.root':
                raise ValueError("Support ROOT file only!")

            # 保存上传的文件
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            logging.info(f"Save file successfully: {filename}")

            json_prefix = os.path.splitext(filename)[0]

            # 使用CommandLogger执行转换命令：root_to_json.py
            cmd_logger.capture_subprocess(
                [
                    'python', './utils/root_to_json.py',
                    '--srcroot', filepath,
                    '--destroot', OUTPUT_FOLDER,
                    '--df_prefix', json_prefix,
                    '--fn_prefix', json_prefix,
                ],
                shell=False
            )

            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 配置生成（添加日志记录）
            json_path = os.path.join(OUTPUT_FOLDER, json_prefix)
            old_config_path = BASE_CONFIG_PATH
            new_config_path = os.path.join(CONFIG_FOLDER, f'new_config_{timestamp}.py')
            logging.debug(f"Start generating configuration ... Original config path: {old_config_path}")

            # 使用CommandLogger执行转换命令：json_to_config.py
            cmd_logger.capture_subprocess(
                [
                    'python', './utils/json_to_config.py',
                    "--json_path", json_path,
                    "--old_config_path", old_config_path,
                    "--new_config_path", new_config_path,
                ],
                shell=False
            )

            logging.info(f"New config path: {new_config_path}")

            pkl_output_path = os.path.join(OUTPUT_FOLDER, 'pkl', f"results_{timestamp}.pkl")
            xlsx_output_path = os.path.join(OUTPUT_FOLDER, f"xlsx/results_{timestamp}.xlsx")

            # 使用CommandLogger执行转换命令：test.py
            cmd_logger.capture_subprocess(
                [
                    'python', '-u',
                    os.path.abspath('./mmdetection-3.3.0/tools/test.py'),
                    os.path.abspath(new_config_path),
                    os.path.abspath(MODEL_WEIGHTS),
                    '--out', os.path.abspath(pkl_output_path),
                ],
                shell=False
            )

            logging.info(f"Test results output path: {pkl_output_path}")

            # 使用CommandLogger执行转换命令：hep_eval.py
            cmd_logger.capture_subprocess(
                [
                    'python',
                    os.path.abspath('./utils/hep_eval.py'),
                    "--pkl_path", os.path.abspath(pkl_output_path),
                    "--json_path", os.path.abspath(json_path),
                    "--output_dir", os.path.abspath(OUTPUT_FOLDER),
                    "--need_excel", str(1),
                    "--excel_name", f"xlsx/results_{timestamp}.xlsx",
                ],
                shell=False
            )

            logging.info(f"Excel file output path: {xlsx_output_path}")

            return render_template('index.html')

        except Exception as e:
            logging.error("ROOT file processing failed!", exc_info=True)
            return render_template('index.html', error=f"ERROR: {str(e)}")

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return render_template('index.html', error="No ROOT file selected")

            # 仅处理ROOT文件（移除文本分支）
            try:
                return handle_root_file(file)
            except Exception as e:
                return render_template('index.html', error=f"ERROR: {str(e)}")

        # 移除文本内容保存分支
        return render_template('index.html', error="Invalid request type")

    # GET请求处理
    return render_template('index.html')


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #


def background_monitor():
    """后台监控任务（添加日志记录）"""
    while True:
        try:
            logging.debug("Background monitor running ...")
            # 示例监控检查
            if not os.path.exists(UPLOAD_FOLDER):
                logging.warning(f"Upload folder not found: {UPLOAD_FOLDER}")
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            time.sleep(60)

        except Exception as e:
            logging.error("Background monitor ERROR", exc_info=True)
            time.sleep(60)

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        safe_dir = os.path.abspath(OUTPUT_FOLDER)
        requested_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, 'xlsx', filename))

        # 路径安全检查
        if not requested_path.startswith(safe_dir):
            app.logger.warning(f"Illegal access path: {filename}")
            return "Forbidden", 403

        if not os.path.exists(requested_path):
            app.logger.error(f"File not found: {filename}")
            return "Not Found", 404

        return send_file(requested_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Download failed: {traceback.format_exc()}")
        return "Server Error", 500

@app.route('/get_logs')
def get_logs():
    """读取app.log日志内容"""
    try:
        log_file = os.path.join(os.getcwd(), 'log/app.log')  # 确保路径与LoggingConfig配置一致

        # 在/get_logs路由中添加调试输出
        print(f"Attempt to read logs: {log_file}")
        with open(log_file, 'r') as f:
             lines = f.readlines()
             print(f"Read {len(lines)} lines from logs")
        return jsonify({"logs": "".join(lines)})

    except FileNotFoundError:
        return jsonify({"error": "Log file not found"}), 404
    except Exception as e:
        logging.error(f"Read logs failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/download_logs')
def download_logs():
    return send_file(
        'app.log',
        as_attachment=True,
        download_name=f'full_log_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        mimetype='text/plain'
    )

def tail(f, lines=500):
    """高效读取文件末尾"""
    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_found = []

    while len(lines_found) < lines and block_end_byte > 0:
        if block_end_byte - BLOCK_SIZE > 0:
            f.seek(block_end_byte - BLOCK_SIZE)
            blocks = f.read(BLOCK_SIZE)
        else:
            f.seek(0)
            blocks = f.read(block_end_byte)
        
        lines_found = blocks.splitlines()
        block_end_byte -= BLOCK_SIZE

    return '\n'.join(lines_found[-lines:])

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/list_xlsx')
def list_xlsx_files():
    """获取XLSX文件列表"""
    files = sorted(
        glob(os.path.join(OUTPUT_FOLDER, 'xlsx/*.xlsx')),
        key=os.path.getmtime,
        reverse=True
    )
    return jsonify([os.path.basename(f) for f in files])

class LogFilter(logging.Filter):
    def filter(self, record):
        # 过滤包含请求方法(GET/POST等)的日志
        if record.args and isinstance(record.args, tuple):
            return not any(
                isinstance(arg, str) and 
                ('GET' in arg or 'POST' in arg)
                for arg in record.args
            )
        return True


if __name__ == '__main__':
    logging.info("Application startup initialization")
    threading.Thread(
        target=background_monitor, 
        daemon=True,
        name="BackgroundMonitor",
    ).start()
    
    # 禁用Werkzeug日志
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    werkzeug_logger.addFilter(LogFilter())
    #werkzeug_logger.disabled = True  # 完全禁用
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logging.critical("Flask application startup failed", exc_info=True)
        sys.exit(1)

