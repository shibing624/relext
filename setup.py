# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages

__version__ = None
exec(open('relext/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='relext',
    version=__version__,
    description='RelExt: A Tool for Relation Extraction from Text.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/relext',
    license="Apache 2.0",
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='relation extraction,relext,relation,extraction',
    install_requires=[
        'loguru',
        'hanlp',
        'tabulate',
        'importlib_metadata',
        'nltk',
        'paddlenlp',
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'relext': 'relext'},
    package_data={'relext': ['*.*', '*.md']}
)
