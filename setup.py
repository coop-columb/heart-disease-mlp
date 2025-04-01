from setuptools import find_packages, setup

setup(
    name="heart-disease-mlp",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "scripts", "docs"]),
)
