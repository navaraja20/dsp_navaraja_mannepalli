from setuptools import setup, find_packages
setup(
    name="house_prices",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'joblib>=1.0.0'
    ],
    python_requires='>=3.8',
)
