import setuptools

setuptools.setup(
    name="transformers",
    version="0.0.1",
    author="ebk",
    author_email="rladmdqh0214@naver.com",
    description="A small example package",
    long_description="aA long example package",
    long_description_content_type="text/markdown",
    url="https://github.com/EungoBoKim/transformers_edited.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)