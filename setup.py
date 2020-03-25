from setuptools import setup

setup(
    name='boffea',
    packages=['brazil'],
    include_package_data=True,
    install_requires=['coffea', 'xxhash', 'scipy', 'mplhep', 'cloudpickle', 'numexpr'],
)
