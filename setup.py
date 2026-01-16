"""
Complexity Diffusion - Mu-Guided Architecture for Image Generation

v0.3.0: Mu-Guided KQV, Mu-Guided Expert Routing, Contextual Mu, Mu-Damped Dynamics
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="complexity-diffusion",
    version="0.3.0",
    description="Mu-Guided DiT with INL Dynamics for image generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pacific Prime",
    author_email="contact@pacific-prime.ai",
    url="https://github.com/Web3-League/complexity-diffusion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.66.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "train": [
            "datasets>=2.16.0",
            "tensorboard>=2.15.0",
            "wandb>=0.16.0",
            "accelerate>=0.25.0",
        ],
        "cuda": [
            "triton>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="diffusion transformer llama inl-dynamics image-generation pytorch",
)
