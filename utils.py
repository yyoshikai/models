def check_leftargs(self, logger, kwargs, show_content=False):
    if len(kwargs) > 0:
        raise ValueError(f"Unknown kwarg in {type(self).__name__}: {list(kwargs.keys())}")

EMPTY = lambda x: x
