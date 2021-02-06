import setuptools

setuptools.setup(
    name="luafun",
    version="0.0.0",
    author="Pierre Delaunay",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'luafun = luafun.main:main',
        ]
    },
    package_data={
        'luafun': [
            'botslua/*.lua',
            'luafun/game/resources/*.json'
        ]
    },
    install_requires=[
        'protobuf',
        'pygtail',
        'torch'
    ],
)
