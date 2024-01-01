from setuptools import setup, find_packages
from typing import List

# Declare variables for setup functions.
PROJECT_NAME = "APS Failure Detection"
VERSION = "0.0.1"
AUTHOR_NAME = "Aritra Ganguly"
AUTHOR_EMAIL = "aritraganguly.msc@protonmail.com"
DESCRIPTION = "Detection of APS Failure at Scania Trucks with Machine Learning."
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements_list() -> List[str]:
    """
    This function returns a list of requirements mentioned in the requirements.txt file.
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [
            requirement_name.replace("\n", "") for requirement_name in requirement_list
        ]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list(),
)
