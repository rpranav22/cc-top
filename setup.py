import setuptools

setuptools.setup(
    name="sscc",
    version="0.0.1",
    author="MM",
    author_email="mm@mm.com",
    description="package for semi-supervised constrained clustering",
    url="foo.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    zip_safe=False
)
