from __future__ import annotations

import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='coco_froc_analysis',
    version='0.0.31',
    author='Alex Olar',
    author_email='olaralex666@gmail.com',
    description='A small package that evaluates COCO detection results from OpenMMLab and Detectron(2).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qbeer/coco-froc-analysis',
    project_urls={
        'Bug Tracker': 'https://github.com/qbeer/coco-froc-analysis/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
