import platform
from os import path as op
import io
from setuptools import setup

here = op.abspath(op.dirname(__file__))
# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")
    if platform.system() == "Windows":
        all_reqs.append("pywin32")

install_requires = [x.strip() for x in all_reqs]

setup(name='veritastool',version='2.0.1',description ='Veritas Diagnosis tool for fairness & transparency assessment.',author='MAS Veritas',
install_requires=install_requires,python_requires='>=3.8.0,<=3.10.10',
packages=['veritastool.model','veritastool.principles','veritastool.metrics','veritastool.config','veritastool.util','veritastool.resources','veritastool.usecases'],
include_package_data=True,
zip_safe=False)
