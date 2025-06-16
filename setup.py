from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup
setup(
    name='JWST_NIRSPecIFUreduction',
    version='1.0',
    description='data reduction and Reference differtial imaging of JWST NIRSpec IFU data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sbetti22/JWST_NIRSPecIFUreduction',
    author='Sarah Betti',
    author_email='sbetti@stsci.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='jwst NIRSpec ifu rdi reduction disk pipeline',
    packages=['JWST_NIRSPecIFUreduction'],
    install_requires=[
        'numpy', 'pathlib', 'jwst',  'scipy', 'astropy', 'matplotlib', 'scikit-image', 'h5py', 'packaging', 'vip_hci'
    ],
    include_package_data=True,
    zip_safe=False
)