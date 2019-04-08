import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from psutil import cpu_count
from honeybadgermpc.config import HbmpcConfig

if HbmpcConfig.extras is None:
    HbmpcConfig.extras = {}

num_workers = HbmpcConfig.extras.get("pc", cpu_count())
executor_type = HbmpcConfig.extras.get("type", "THREAD")
logging.info("%s POOL EXECUTOR MAX WORKERS: %d",  executor_type, num_workers)
executor = ProcessPoolExecutor(
    max_workers=num_workers) if executor_type != "THREAD" else ThreadPoolExecutor(
        max_workers=num_workers)
