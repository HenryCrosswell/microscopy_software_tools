from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="microscopy-analysis-tools",
    version="0.1",
    description="Automated analysis tools for lightsheet and two-photon microscopy data",
    author="Henry Crosswell",
    url="https://github.com/HenryCrosswell/microscopy_software_tools",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

