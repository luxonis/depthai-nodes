from setuptools import find_packages, setup

setup(
    name="depthai-nodes",
    version="0.1.0",
    description="Python library for on-host depthai augmentation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Luxonis",
    author_email="support@luxonis.com",
    url="https://github.com/luxonis/depthai-nodes",
    packages=find_packages(),
    install_requires=[
        "depthai",
        # Other dependencies
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # TODO
)
