import sys


def error_message_detail(exception: Exception, error_detail: sys) -> str:
    """
    Generates a detailed error message that includes:
        - The file path where the exception occurred,
        - The line number in that file, and
        - The error message from the exception.

    :param exception: The exception instance that was captured during error handling.
    :param error_detail: The sys module, which is used here to access the traceback information.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Script Path: [{file_name}]; Line Number: [{line_number}]; Error Message: [{exception}]"


class SensorException(Exception):
    """
    Custom Exception class that extends the built-in Exception.

    This class formats error messages to include file name and line number details.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
