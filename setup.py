from setuptools import setup

setup(
    name='boffea',
    packages=['brazil'],
    include_package_data=True,
    install_requires=['numpy==1.22.0', 'mplhep==0.1.10', 'uproot_methods==0.7.3', 'uproot==3.11.3', 'awkward==0.12.20', 'matplotlib==3.2.1', 'coffea==0.6.37', 'xxhash', 'scipy', 'mplhep', 'cloudpickle', 'numexpr'],
)
