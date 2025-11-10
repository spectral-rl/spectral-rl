import os
import pathlib

from setuptools import find_packages, setup


def get_version() -> str:
    """Extract version from __init__.py file."""
    init_path = pathlib.Path("spectralrl") / "__init__.py"
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read().split()
    return content[content.index("__version__") + 2][1:-1]


def get_install_requires() -> list[str]:
    """Return the list of required packages for installation."""
    return [
        # Core dependencies
        "numpy>=1.21.0",
        "torch>=1.8.0",
        "torchvision>=0.9.0",

        # RL environments
        "gym>=0.23.1,<0.24.0",
        "gymnasium>=1.0.0",
        "dm_control==1.0.27",
        "dm_env==1.6",
        "mujoco==3.2.7",
        "mujoco-py==2.1.2.14",

        # Configuration and logging
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",

        # Utilities
        "tqdm>=4.64.0",
        "h5py>=3.6.0",
        "imageio>=2.19.0",
        "Cython==0.29.36",

        # Development tools
        "pre-commit>=2.20.0",
    ]

# Project metadata
ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION = get_version()

# Setup configuration
setup(
    name="spectralrl",
    version=VERSION,
    author="Spectral-RL Team",
    author_email="cgao@gatech.edu",
    description="Advanced spectral representation learning for reinforcement learning and causal inference",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rep-rl/spectral-rl",
    license="MIT",
    keywords="reinforcement learning, representation learning, spectral methods, causal inference",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=get_install_requires(),
)
