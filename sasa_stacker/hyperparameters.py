MODEL_INPUTS = 160
NUMBER_OF_WAVLENGTHS = MODEL_INPUTS
WAVLENGTH_START = 0.4
WAVLENGTH_STOP = 1.2
MODEL_DISCRETE_OUTPUTS = 8
MODEL_CONTINUOUS_OUTPUTS = 10
MODEL_DISCRETE_PREDICTIONS = {
    "particle_material" : ["Au", "Al"],
    "hole" : ["holes", "no holes"]
    }

BATCH_SIZE = 128
EPOCHS = 10
MOMENTUM = 0.95
INIT_LR = 2e-3
# Important!
# This determine which interval of wavlengths should be used for the fit
FIT_BOUNDS = (0.65, 0.95)
