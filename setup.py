from setuptools import setup, find_packages


setup(
        name='pso_rff',
        version='0.1.0',
        packages=find_packages('pso', 'rff'),
        install_requires=['numpy', 'matplotlib']
        )

