from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit('Bionlp requires Python 3.')

with open('requirements.txt') as f:
    reqs = f.read()

setup(
        name='bionlp',
        version='0.0.1',
        description='tools for extraction adverse drug events and relations with medications',
        url='https://github.com/kearnsw/bio-nlp',
        license='GPLv3.0',
        packages=find_packages(exclude=('notebooks')),
        install_requires=reqs.strip().split('\n'),

)
