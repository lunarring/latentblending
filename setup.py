from setuptools import setup, find_packages

# Read requirements.txt and store its contents in a list
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='latentblending',
    version='0.2',
    url='https://github.com/lunarring/latentblending',
    description='Butter-smooth video transitions',
    long_description=open('README.md').read(),
    install_requires=required,
    dependency_links=[
        'git+https://github.com/lunarring/lunar_tools#egg=lunar_tools'
    ],
    include_package_data=False,
)

