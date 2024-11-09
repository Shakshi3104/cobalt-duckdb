from functools import wraps
import time

from loguru import logger


# https://qiita.com/hisatoshi/items/7354c76a4412dffc4fd7
def stop_watch(func):
    """
    処理にかかる時間計測をするデコレータ
    """
    @wraps(func)
    def wrapper(*args, **kargs):
        logger.debug(f"🚦 [@stop_watch] measure time to run `{func.__name__}`.")
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        logger.debug(f"🚦 [@stop_watch] take {elapsed_time:.3f} sec to run `{func.__name__}`.")
        return result
    return wrapper