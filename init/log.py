import logging, re, os

def setup_log(log_name):
    """
    建立並設定一個每日輪轉的 logger（每日自動建立新檔案，並保留一定數量的歷史檔案）。

    Parameters:
    log_name (str): 日誌檔案名稱（可含路徑），例如 "../log/faceLog"

    Returns:
    logger (logging.Logger): 已設定好檔案輪轉與格式的 logger 物件
    """

    # 建立 logger 物件
    logger = logging.getLogger(log_name)
    log_path = os.path.join(os.path.dirname(__file__), log_name)

    # 如果 log 目錄不存在則建立
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "../log")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "../log"))

    # 設定 logger 等級為 INFO
    logger.setLevel(logging.INFO)

    # 建立時間輪轉的檔案處理器
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path,               # 實際寫入檔案的路徑
        when="MIDNIGHT",                # 每天午夜輪轉一次
        interval=1,                     # 間隔為 1 天
        backupCount=20,                 # 最多保留 20 個歷史檔案
        encoding='utf-8'                # 編碼格式為 UTF-8
    )

    # 檔案後綴格式（例如：faceLog.2025-05-20.log）
    file_handler.suffix = "%Y-%m-%d.log"

    # 使用正規表達式配對後綴（確保符合後綴才會刪除）
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")

    # 設定輸出格式
    file_handler.setFormatter(
        logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(module)s] [%(filename)s:%(lineno)s] %(message)s'
        )
    )

    # 將處理器加入 logger
    logger.addHandler(file_handler)

    return logger


LOGGER = setup_log("../log/faceLog")
