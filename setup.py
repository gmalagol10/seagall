from setuptools import setup, find_packages

setup(
    name="seagall",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Gabriele",
    author_email="gabriele.malagoli3@gmail.com",
    description="Single-cell explain etc etc",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gmalagol10/seagall.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
