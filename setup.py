"""Create instructions to build Cryo Tools"""
import setuptools

requirements = []

setuptools.setup(
    name="cryo_em_SBI",
    maintainer=[
        "David Silva-Sánchez",
        "Sonya Hanson",
        "Pilar Cossio",
    ],
    version="0.0.1",
    maintainer_email=[
        "david.silva@yale.edu",
    ],
    description="Simple module to do cryo-EM related things (CTF, Noise, generate synthetic data) for people new to the field",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DSilva27/Cryo_Tools.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
