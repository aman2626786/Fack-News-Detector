from setuptools import setup, find_packages

setup(
    name="fake_news_detector",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "gunicorn==21.2.0",
        "joblib==1.3.2",
        "numpy==1.24.3",
        "pandas==1.5.3",
        "python-dateutil==2.9.0.post0",
        "scikit-learn==1.3.0",
        "scipy==1.15.2",
        "threadpoolctl==3.6.0"
    ],
)
