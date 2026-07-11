from setuptools import setup

setup(
  name="lambda-labs",
  version="1.0.1",
  description="Small CLI for Lambda Cloud.",
  py_modules=["lambda_labs"],
  install_requires=["rich"],
  entry_points={"console_scripts": ["lambda-labs=lambda_labs:main"]},
  python_requires=">=3.11",
)
