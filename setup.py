import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(

    name='sasa_stacker',

    version='0.1',

    author="Tim Luca Turan",

    author_email="timturan@web.de",

    description="Find a meta surface stack to a target spectrum",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/TimLucaTuran/stacker",

    packages=['sasa_stacker'],

    install_requires=requirements,

    license='MIT',

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

)
