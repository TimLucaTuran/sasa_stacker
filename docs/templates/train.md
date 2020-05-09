## Usage
<pre><code>
usage: train.py [-h] [-p PARAMS] [-s S_MATS] [-log LOG_DIR] [-n]
                [-mt MODEL_TYPE] [-f FORWARD_MODEL] [-i INVERSE_MODEL]
                [-db DATABASE]
                m b

positional arguments:
  m                     path to output model
  b                     path to directory containing the training batches

optional arguments:
  -h, --help            show this help message and exit
  -p PARAMS, --params PARAMS
                        path to the .pickle file containing the smat
                        parameters
  -s S_MATS, --s-mats S_MATS
                        path to the directory containing the smats
  -log LOG_DIR, --log-dir LOG_DIR
                        path to dir where the logs are saved
  -n, --new             train a new model
  -mt MODEL_TYPE, --model-type MODEL_TYPE
                        ["inverse", "forward", "combined"] which kind of model
                        to train
  -f FORWARD_MODEL, --forward-model FORWARD_MODEL
                        needs to be provided when training a combined model
  -i INVERSE_MODEL, --inverse-model INVERSE_MODEL
                        needs to be provided when training a combined model
  -db DATABASE, --database DATABASE
                        sqlite database containing the adresses
</code></pre>