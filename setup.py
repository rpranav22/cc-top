import setuptools

setuptools.setup(
    name="cctop",
    version="0.0.1",
    author="MM",
    author_email="anon@anonymous.com",
    description="package for the cctop project",
    url="cctop.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    zip_safe=False
)
