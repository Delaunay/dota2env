import setuptools
# from torch.utils import cpp_extension


setuptools.setup(
    name="luafun",
    version="0.0.0",
    author="Pierre Delaunay",
    packages=setuptools.find_packages(),
    # ext_modules=[
    #     cpp_extension.CppExtension('mmcust', ['luafun/cpp/mmcust.cpp'])
    # ],
    # cmdclass={
    #     'build_ext': cpp_extension.BuildExtension
    # },
    entry_points={
        'console_scripts': [
            'luafun = luafun.main:main',
            'luafun-drafter = luafun.drafter:main',
            'luafun-draft-extractor = luafun.steamapi.steamapi:main',
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
