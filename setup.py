from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='latentblending',
    version='0.1',
    install_requires=requirements,
)
