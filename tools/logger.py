import sys, logging
from tqdm import tqdm
import inspect

log_name2level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
    "%y%m%d %H:%M:%S")

class TqdmHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            sys.stdout.flush()
            self.flush()
        except Exception:
            self.handleError(record)

default_loggers = {}

def default_logger(filename=None, stream_level='info', 
        file_level='debug', logger_name=None, nprocess=None, iprocess=None):
    if logger_name is None:
        mod = inspect.stack()[1]
        logger_name = inspect.getmodulename(mod.filename)+'.'+mod.function
    logger = logging.getLogger(logger_name)

    if nprocess is None:
        assert iprocess is None
        formatter = logging.Formatter(f"[%(asctime)s][%(levelname)s] %(message)s", "%y%m%d %H:%M:%S")
    else:
        assert iprocess is not None
        iprocess = str(iprocess).rjust(len(str(nprocess)), '0')
        formatter = logging.Formatter(f"[%(asctime)s][{iprocess}/{nprocess}][%(levelname)s] %(message)s", "%y%m%d %H:%M:%S")
    if logger_name not in default_loggers:
        default_loggers[logger_name] = logger
        stream_handler = TqdmHandler(level=log_name2level[stream_level])
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if filename is not None:
            if not isinstance(filename, list):
                filename = [filename]
            for filename0 in filename:
                file_handler = logging.FileHandler(filename=filename0)
                file_handler.setLevel(log_name2level[file_level])
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info(f"PYTHON: {sys.version}")
    try:
        import torch
        logger.info(f"PYTORCH: {torch.__version__}")
    except ModuleNotFoundError:
        logger.info(f"PyTorch not installed.")

    return logger

def log_config(logger: logging.Logger, config, level=logging.INFO, prefix=''):
    if isinstance(config, list):
        config = {f'[{i}]': value for i, value in enumerate(config)}
    for key, value in config.items():
        if isinstance(value, (list, dict)):
            logger.log(level, prefix+f"{key}:")
            log_config(logger, value, level, prefix=prefix+"  ")
        else:
            logger.log(level, prefix+f"{key}: {value}")
