from setuptools import setup, find_packages

setup(
    name="dynaroute",
    version="0.2.0",
    description="Dynamic SSM-Transformer Routing for Efficient Sequence Modeling",
    author="Research Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "isort", "mypy"],
        "visualization": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "tracking": ["wandb>=0.15.0", "tqdm>=4.65.0"],
    },
)
