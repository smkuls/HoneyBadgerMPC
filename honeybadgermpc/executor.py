import logging
from concurrent.futures import ProcessPoolExecutor
from psutil import cpu_count
from honeybadgermpc.config import HbmpcConfig


num_workers = HbmpcConfig.extras.get("pc", cpu_count())
logging.info("PROCESS POOL EXECUTOR MAX WORKERS: %d", num_workers)
executor = ProcessPoolExecutor(max_workers=num_workers)
