from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="keras_tuner_cv",
    version="0.0.2",
    description="Extension for keras tuner that adds a set of classes to implement cross validation techniques.",
    license="GPL v3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files="LICENSE",
    dec="README.md",
    author="Giuseppe Grieco",
    author_email="g.grieco1997@gmail.com",
    url="https://github.com/giuseppegrieco/keras-tuner-cv",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["keras_tuner"],
)
