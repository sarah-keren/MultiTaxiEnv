# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:22:44 2020

@author: Kevin
"""

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='MultiTaxiEnv',
    url='https://github.com/sarah-keren/MutliTaxiEnv',
    author='Sarah Keren, Howie Guo, Kevin Huang',
    # Needed to actually package something
    packages=['multitaxienv'],
    # Needed for dependencies
    install_requires=['numpy','gym'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='None',
    description='Multiple Enviroments',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)