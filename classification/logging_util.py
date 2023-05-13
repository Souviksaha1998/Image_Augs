import logging
import os
from time import strftime
from typing import List, Union, TypeVar


time_now = strftime("%d-%m-%Y")

WARNING = TypeVar("WARNING")
ERROR = TypeVar("ERROR")
INFO = TypeVar("INFO")
DEBUG = TypeVar("DEBUG")


def get_logger(base_name: os.path.basename(__file__)) -> logging.Logger:

    """
    Get_logger will take python script name (base_name) Ex : os.path.basename(__file__) , and mark it as base_name for formatter.
    It will be helpful for Debugging.
    Get logger will create a log folder & save the logs_(%d%M%Y) there.

    : param base_name : base_name takes python script name , Ex : os.path.basename(__file__).split('.')[0] --> logging.py --> logging

    """

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)
    format = logging.Formatter(
        f"%(asctime)s - %(levelname)s - {base_name} - line - %(lineno)d ==> %(message)s"
    )
    file_name = logging.FileHandler(f"logs/log_{time_now}.log", mode="a", delay=False)
    file_name.setFormatter(format)
    logger.addHandler(file_name)

    return logger


def log_file_reader(
    file_name: os.PathLike, level_name: Union[WARNING, ERROR, INFO, DEBUG] = WARNING
) -> List:

    """
    Log file reader will reads log files according to log_level_name , default set to 'WARNING'.

    : param log_file_name : Expects .log format file path.
    : param level name    : Log message show according to its level name , default 'WARNING'.
    : return              : It will return list of specific log messages.


    """

    assert file_name.endswith(".log"), "please provide correct format... (.log) file"

    with open(file_name, "r") as file_reader:
        reads = file_reader.readlines()
        filter_log = [line for line in reads if level_name in line]
        return filter_log


# if __name__ == "__main__":
#     reads = log_file_reader("logs/log_25-08-2022.log", "INFO")
#     for t in reads:
#         print(t)
