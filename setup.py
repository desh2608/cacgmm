from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cacgmm-gpu",
    version="0.1.0",
    description="GPU-based CACGMM trainer in CuPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desh2608/cacgmm",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    keywords="speech enhancement gss",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
)
