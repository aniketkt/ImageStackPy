#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:56:50 2018

@author: atekawade
"""

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ImageStackPy',
    url='https://github.com/aniketkt/ImageStackPy',
    author='Aniket Tekawade',
    author_email='aniketkt@gmail.com',
    # Needed to actually package something
    packages=['ImageStackPy'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='none',
    description='Fast, parallelized image processing and viewing of large greyscale image stacks. Includes an object tracking toolkit',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
