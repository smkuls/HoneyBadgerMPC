from concurrent.futures import ThreadPoolExecutor
from psutil import cpu_count

executor = ThreadPoolExecutor(max_workers=cpu_count())
