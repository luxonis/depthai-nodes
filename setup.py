from setuptools import setup, find_packages

setup(
    name='depthai-nodes',
    version='0.1.0',
    description='Python library for on-host depthai augmentation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Luxonis',
    author_email='your.email@example.com',
    url='https://github.com/luxonis/depthai-nodes',
    packages=find_packages(),
    install_requires=[
        'depthai',
        # Other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # TODO
)