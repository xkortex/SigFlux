from setuptools import setup, find_packages

# https://python-packaging.readthedocs.org/en/latest/dependencies.html
# https://gehrcke.de/2014/02/distributing-a-python-command-line-application/
# python setup.py install develop

setup(name='sigflux',
      version='0.1.0',
      description='Algorithms for processing signals',
      url='https://github.com:xkortex/SigFlux.git',
      author='Michael McDermott',
      author_email='mikemcdermott23@gmail.com',
      license='Private',
      packages=find_packages(exclude=["*.testing", "testing.*", "testing", "*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=[
          'matplotlib==2.0.0',
          'numpy==1.12.1',
          'scipy==0.19.0',
          'pandas==0.19.2',
          'PyWavelets==0.5.2',
      ],
      zip_safe=False)
