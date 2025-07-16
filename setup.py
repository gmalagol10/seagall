from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call([
            'pip', 'install', 
            'torch-scatter', 
            'torch-sparse', 
            '--find-links', 
            'https://data.pyg.org/whl/torch-2.6.0+cpu.html'
        ])
        install.run(self)

setup(
    name="seagall",
    version="0.1",
    packages=find_packages(),
    install_requires=[l.strip() for l in open('requirements.txt').readlines()],
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
    cmdclass={
        'install': CustomInstallCommand,
    },
)
