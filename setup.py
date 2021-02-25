import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mggp", # Replace with your own username
    version="0.0.0",
    author="Henrique Castro",
    author_email="henriquec.castro@outlook.com",
    description="Multi Gene Genetic Programming toolbox for System Identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['deap'],
)