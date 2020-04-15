# SASA Stacker
![Flowtchart](https://raw.githubusercontent.com/TimLucaTuran/bachlor-arbeit/master/fig/al_algo.svg?sanitize=true)

Documentation: https://timlucaturan.github.io/sasa_stacker/

## Installation
This package depends on  googles `tensorflow`, which at the time of writing only supports python 3.5 - 3.7, so I recommend creating a virtual environment:

`$ python3.7 -m venv sasa-venv`

and activating it

`$ source sasa-venv/bin/activate`

Now clone this repository

`$ git clone https://github.com/TimLucaTuran/sasa_stacker`

cd into the repository and install the package

`$ cd sasa_stacker`

`$ pip install .`

Test if everything worked

`$ cd sasa_stacker`

`$ python fit.py data/test.npy`
