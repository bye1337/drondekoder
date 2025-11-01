"""Setup script for Drone Visual Positioning System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="drondekoder2.0",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Система визуального позиционирования дрона на основе компьютерного зрения",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drondekoder2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "drondekoder-train=src.training.train:main",
            "drondekoder-test=scripts.test_model:main",
            "drondekoder-create-dataset=scripts.create_dataset:main",
        ],
    },
)

