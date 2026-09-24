from setuptools import setup, find_packages

setup(
    name="hastejev",
    version="1.1.0",
    packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "scripts", "docs"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "transformers>=4.30.0",
        ],
    },
    author="hastejev Team",
    description="Ultra-Low-Latency System-1 AI Decision Engine",
    long_description=open("README.pypi.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/racstan/hastejev",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
