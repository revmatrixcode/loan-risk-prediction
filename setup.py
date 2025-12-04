from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loan-risk-predictor",
    version="1.0.0",
    author="shanujansh@gmail.com",
    author_email="shanujansh@gmail.com",
    description="A machine learning package for predicting loan default risk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shanujans/loan-risk-prediction",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "loan-risk-predict=loan_risk_predictor.predict:main",
        ],
    },
    include_package_data=True,
)