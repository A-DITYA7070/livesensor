from setuptools import find_packages,setup
from typing import list

def get_requirements()-> list[str]:
    requirements_list : list[str]=[]
    return requirements_list


setup(
    name="SensorDetection",
    version="0.0.1",
    author="Aditya Raj",
    author_email="adityaraj993148@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements(),
)

