def check_leftargs(self, logger, kwargs):
    if len(kwargs) > 0 and logger is not None:
        logger.warning(f"Unknown kwarg in {type(self).__name__}: {kwargs}")

EMPTY = lambda x: x
