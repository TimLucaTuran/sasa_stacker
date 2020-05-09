<pre><code>
    usage: data_gen.py [-h] [-p PARAMS] [-n NUMBER_OF_BATCHES] [-db DATABASE]
                    src dst

    positional arguments:
    src                   path to source directory containing .npy files
    dst                   path to destination batch directory

    optional arguments:
    -h, --help            show this help message and exit
    -p PARAMS, --params PARAMS
                            path to the .pickle file containing the smat
                            parameters
    -n NUMBER_OF_BATCHES, --number-of-batches NUMBER_OF_BATCHES
    -db DATABASE, --database DATABASE
                            sqlite database containing the adresses
</code></pre>