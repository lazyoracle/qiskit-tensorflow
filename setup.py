from setuptools import setup

setup(
    name="qiskit-tensorflow",
    version="0.0.1",
    description="A Tensorflow based simulator backend for Qiskit",
    url="http://github.com/lazyoracle/qiskit-tensorflow",
    author="Anurag Saha Roy",
    author_email="contact@anuragsaharoy.me",
    include_package_data=True,
    packages=[
        "qiskit_tensorflow",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "qiskit==0.23.*",
        "tensorflow==2.4.*",
    ],
    python_requires=">=3.6",
)
