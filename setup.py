from setuptools import setup, find_packages

# Parse requirements from requirements.txt
requirements = []
for line in open("requirements.txt"):
    if line and not line.startswith("#"):
        requirements.append(line)

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="keras_tuner_cv",
    version="1.0.0",
    description="Extension for keras tuner that adds a set of classes to implement cross validation techniques.",
    license="Apache License 2.0",
    long_description=long_description,
    author="Giuseppe Grieco",
    author_email="g.grieco1997@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
)
