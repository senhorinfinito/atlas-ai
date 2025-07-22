from setuptools import setup, find_packages

setup(
    name='atlas-ai',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'atlas = atlas.cli:main',
        ],
    },
    install_requires=[
        'pylance',
        'pyarrow',
        'pandas',
        'click',
        'Pillow',
        'psutil',
        'matplotlib',
        'pyyaml',
        'pycocotools',
        'datasets',
        'torchcodec',
        'lancedb',
        'tqdm',
        'rich'
    ],
    extras_require={
        'audio': ['soundfile', 'transformers', 'torch']
    },
)
