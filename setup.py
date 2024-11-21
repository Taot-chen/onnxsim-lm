# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
    ["src"]

package_data = \
    {"": ["*"]}

install_requires = \
    [
        "onnx>=1.17.0,<2.0.0",
        "onnxsim>=0.4.36,<0.5.0"
    ]

entry_points = \
{'console_scripts': ['onnxsim-lm = src.main:run']}

setup_kwargs = {
    'name': 'onnxsim-lm',
    'version': '0.1.0',
    'description': 'Work in progress...',
    'long_description': None,
    'author': 'oehuosi',
    'author_email': 'oehuosi@qq.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/Taot-chen/onnxsim-lm',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
