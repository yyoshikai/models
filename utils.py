import itertools

EMPTY = lambda x: x

def get_set(config):
    if isinstance(config, list):
        return config
    elif isinstance(config, dict):
        if 'stop' in config:
            return range(config.get('start', 0), config['stop'], config.get('step', 1))
        else:
            return itertools.count(config.get('start', 0), config.get('step', 1))
    else:
        raise ValueError

