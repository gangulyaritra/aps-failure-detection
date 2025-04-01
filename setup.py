from setuptools import find_packages, setup

with open("./requirements/requirements.txt", "r") as req_file:
    REQUIREMENTS = req_file.read().splitlines()

setup(
    name="aps-failure-detection",
    version="1.0.0",
    author="Aritra Ganguly",
    author_email="aritraganguly.in@protonmail.com",
    description="Detection of APS Failure at Scania Trucks with Machine Learning.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gangulyaritra/aps-failure-detection.git",
    license="MIT",
    keywords="Anomaly Detection, Machine Learning, APS Failure at Scania Trucks",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=REQUIREMENTS,
    package_data={"": ["*"]},
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "run_pipeline = aps.main:main",
            "run_load_data = aps.load_data.etl:main",
        ]
    },
)
