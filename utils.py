import time
import logging

def generate_logger(title):
    ctime = time.asctime( time.localtime(time.time()) )
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig( level=logging.DEBUG,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(f'./log/{title}_{ctime}.log')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger=logging.getLogger()

    return logger