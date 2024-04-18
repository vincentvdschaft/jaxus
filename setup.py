from pathlib import Path

from setuptools import find_packages, setup

requirements_path = Path(__file__).parent / "requirements.txt"

with open(requirements_path) as f:
    requirements = f.read().splitlines()


setup(
    name="jaxus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    package_data={
        # If any package contains *.mplstyle files, include them:
        '': ['*.mplstyle'],
        'jaxus': ['styles/*.mplstyle'],
    },
)
