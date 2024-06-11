# 何もloadしなくてよいもののみ
import time
from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield None

def prog(marker='*'):
    print(marker, flush=True, end='')

class Timer:
    def __init__(self, output=print):
        self.output = output
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.output(time.time() - self.start)