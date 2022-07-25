from setuptools import setup
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('mtsgdst/schema_guided_dst')

setup(
   name='mtsgdst',
   version='1.0',
   author='Lefteris Kapelonis',
   author_email='lkapelonis@gmail.com',
   packages=['mtsgdst', 'mtsgdst.data'],
   package_data={'mtsgdst': extra_files},
   install_requires=[
    'absl-py>=0.7.0',
    'fuzzywuzzy>=0.17.0',
    'numpy>=1.16.1',
    'six>=1.12.0',
    'tensorflow>=2.6.3',
    'transformers==4.18.0',
    'torch==1.11.0',
    'tqdm',
    'scikit-learn',
    'matplotlib',
    'nltk',
   ]
)
