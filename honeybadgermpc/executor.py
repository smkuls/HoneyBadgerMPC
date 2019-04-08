import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from psutil import cpu_count
from honeybadgermpc.config import HbmpcConfig


num_workers = HbmpcConfig.extras.get("pc", cpu_count())
executor_type = HbmpcConfig.extras.get("type", "THREAD")
logging.info("%s POOL EXECUTOR MAX WORKERS: %d",  executor_type, num_workers)
executor = ProcessPoolExecutor(
    max_workers=num_workers) if executor_type == "PROCESS" else ThreadPoolExecutor(
        max_workers=num_workers)
