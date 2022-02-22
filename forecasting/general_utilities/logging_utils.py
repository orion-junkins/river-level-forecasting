import sys
import logging
def build_logger(log_level='INFO', name='NewLogger', log_file='logfile.log'):
    logger = logging.getLogger(name=name)
    fileHandler = logging.FileHandler(log_file)
    streamHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pathname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(log_level)
    return logger
