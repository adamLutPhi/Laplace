from distutils.core import setup

with open('README') as file: 
    readme = file.read()

setup(
    name = 'Laplace'
    version = '1.0.1',
    packages = ['Laplace'],
    url = 'https://github.com/adamwillisMastery/Laplace',
    license='License.md',
    description='Laplace ',
    long_description='Readme.md'
    author = 'Ahmad Lutfi',
    author_email = 'obzajd@gmail.com'
    )