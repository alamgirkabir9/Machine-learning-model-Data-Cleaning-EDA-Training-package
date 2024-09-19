from setuptools import setup, find_packages

setup(
    name='cleantransformer',  
    version='0.1.0',  
    author='Alamgir kabir',
    author_email='alomgirkabir720@gmail.com',
    description='A package for data cleaning, transformation, and model evaluation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/cleantransformer_package',  
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
