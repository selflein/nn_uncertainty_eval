import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uncertainty_eval",
    version="0.0.1",
    author="Sven Elflein",
    author_email="elflein.sven@googlemail.com",
    description="Uncertainty Estimation Evaluation Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/selflein/nn_uncertainty_eval",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6.0",
        "torchvision>=0.8.1",
        "numpy>=1.18.1",
        "scikit-learn>=0.22.1",
        "matplotlib>=3.1.3",
        "tfrecord==1.11",
    ],
)
