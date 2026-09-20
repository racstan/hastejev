from setuptools import setup, find_packages

setup(
    name="hastejev",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "transformers>=4.30.0",
            "huggingface-hub>=0.20.0",
        ],
    },
    author="hastejev Team",
    description="Ultra-Low-Latency, Zero-Copy System-1 AI Decision Engine",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/rachitasthana/hastejev",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
