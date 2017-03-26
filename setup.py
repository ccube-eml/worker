from setuptools import setup, find_packages

requirements = [
    'click',
    'requests',
    'pika',
    'jprops',
    'sklearn',
    'numpy',
    'scipy',
]

setup(
    name='worker',
    version='',
    url='',
    license='',
    author='John Doe',
    author_email='john.doe@ccube.com',
    description='',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': ['ccube-worker=worker.__main__:cli'],
    }
)
