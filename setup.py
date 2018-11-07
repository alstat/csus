from setuptools import setup, find_packages

setup(
    name='csus',
    version='0.0.1',
    description='Cross-Sell and Upsell',
    author='jonats, jp, al',
    url='https://github.com/alstat/csus',
    packages=['api'],
    install_requires=[
        'unidecode',
        'implicit',
        'numpy',
        'pandas',
        'random',
        'scipy',
        'sklearn'
    ]
)