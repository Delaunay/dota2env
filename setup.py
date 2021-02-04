import setuptools

setuptools.setup(
    name="luafun",
    version="0.0.0",
    author="Pierre Delaunay",
    packages=setuptools.find_packages(),
    install_requires=[
        'protobuf',
        'pygtail',
        'torch'
    ],
)
