import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tmps",
    version="0.1.dev2",
    author="Logan Hillberry",
    author_email="lhillberry@gmail.com",
    description="thermo-magnetic particle simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lhillber/tmps",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
