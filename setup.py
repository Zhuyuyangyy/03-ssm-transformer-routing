from setuptools import setup, find_packages

setup(
    name="dynaroute",
    version="0.1.0",
    description="Dynamic SSM-Transformer Routing for Efficient Sequence Modeling",
    author="Research Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy"],
        "visualization": ["matplotlib", "seaborn"],
    },
)
