from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="avhubert-hf",
    version="0.1.0",
    author="Meta AI Research (Refactored)",
    description="Audio-Visual HuBERT for self-supervised speech representation learning (HuggingFace style)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/av_hubert",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scipy>=1.5.0",
        "sentencepiece>=0.1.96",
        "editdistance>=0.6.0",
        "python-speech-features>=0.6",
        "omegaconf>=2.1.0",
        "hydra-core>=1.1.0",
        "tqdm>=4.60.0",
        "librosa>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
)