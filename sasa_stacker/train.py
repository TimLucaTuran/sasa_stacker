#standard lib modules
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sqlite3
import argparse
import pickle
import cProfile
import matplotlib

#NN modules
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import CustomObjectScope, Progbar
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

#Self written modules
from sasa_db.crawler import Crawler
from sasa_phys.stack import *
from hyperparameters import *
from custom_layers import avg_init, RunningAvg, ZeroPadding1DStride2, load_inverse_from_combined
from data_gen import create_random_stack
#%%
def create_inverse_model():
    inp = Input(shape=(MODEL_INPUTS, 2))
    x = Conv1D(16, 5, activation='relu')(inp)
    x = MaxPooling1D()(x)
    x = Conv1D(32, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Conv1D(64, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    conv_out = GlobalMaxPooling1D()(x)

    #discrete branch
    x = Dense(128, activation='relu')(conv_out)
    x = Dropout(0.3)(x)
    c1 = Dense(2, activation='softmax')(x)
    c2 = Dense(2, activation='softmax')(x)
    c3 = Dense(2, activation='softmax')(x)
    c4 = Dense(2, activation='softmax')(x)
    discrete_out = Concatenate(name='discrete_out')([c1, c2, c3, c4])

    #continuous branch
    x = Dense(128, activation='relu')(conv_out)
#    x = BatchNormalization()(x)
    continuous_out = Dense(MODEL_CONTINUOUS_OUTPUTS, activation='linear', name='continuous_out')(x)

    model = Model(inputs=inp, outputs=[discrete_out, continuous_out])
    print(model.summary())
    return model

def create_forward_model():
    #merge the output of the inverse network
    dis_in = Input(shape=MODEL_DISCRETE_OUTPUTS)
    cont_in = Input(shape=MODEL_CONTINUOUS_OUTPUTS)
    x = Concatenate()([dis_in, cont_in])
    x = BatchNormalization(momentum=MOMENTUM)(x)

    x = Dense(21*128)(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Reshape((21,128))(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = UpSampling1D()(x) #40,128

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = RunningAvg(64, 3, padding='same')(x)
    x = UpSampling1D()(x) #80,128

    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = RunningAvg(32, 3, padding='same')(x)
    x = UpSampling1D()(x) #160,64

    x = Conv1D(2, 3, activation='linear', padding='same')(x) #160,2
#    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = RunningAvg(2, 5, padding='valid')(x)
    x = RunningAvg(2, 5, padding='valid')(x)
#    x = RunningAvg(2, 5)(x)
#    x = RunningAvg(2, 3)(x)
    model = Model(inputs=[dis_in, cont_in], outputs=x)
    return model

class LossWeightsChanger(tf.keras.callbacks.Callback):
    def __init__(self, continuous_out_loss):
        self.continuous_out_loss = continuous_out_loss

    def on_epoch_end(self, epoch, logs={}):
        print("[INFO] current weight:", self.continuous_out_loss)
        print("[INFO] discrete_loss:", logs["discrete_out_loss"])
        print("[INFO] discrete_loss:", logs["continuous_out_loss"])
        self.continuous_out_loss = (self.continuous_out_loss *
            logs["discrete_out_loss"]/logs["continuous_out_loss"])

class BatchUpdater(tf.keras.callbacks.Callback):
    def __init__(self, batch_X):
        self.batch_X = batch_X

    def on_batch_begin(self, batch, logs={}):
        print("[INFO] batch[0].shape", batch[0].shape)
        self.batch_X = batch[0]

def mse_with_changable_weight(loss_weight):
    def loss(y_true, y_pred):
        loss_val = mean_squared_error(y_true, y_pred)
        return loss_weight*loss_val

    return loss

def batch_generator(batch_dir):
    """
    Just load the batches created by data_gen.py
    """

    inp_batches = []
    while True:
        #reset x_batches once are batches are used up
        if len(inp_batches) == 0:
            inp_batches = os.listdir(f"{batch_dir}/X")

        idx = random.randint(0, len(inp_batches)-1)
        batch = inp_batches[idx][:-4]#[:-4] to remove the .npy

        x = np.load(f"{batch_dir}/X/{batch}.npy")
        discrete_out = np.load(f"{batch_dir}/Y/{batch}.npy")

        with open(f"{batch_dir}/params/{batch}.pickle", "rb") as f:
            params = pickle.load(f)
        continuous_out = np.zeros((BATCH_SIZE, MODEL_CONTINUOUS_OUTPUTS))

        for i in range(BATCH_SIZE): #needs to be generalized
            layer1, layer2, stack = params[i]

            continuous_out[i,0] = layer1["width"]
            continuous_out[i,1] = layer1["length"]
            continuous_out[i,2] = layer1["thickness"]
            continuous_out[i,3] = layer1["periode"]

            continuous_out[i,4] = layer2["width"]
            continuous_out[i,5] = layer2["length"]
            continuous_out[i,6] = layer2["thickness"]
            continuous_out[i,7] = layer2["periode"]

            continuous_out[i,8] = stack["spacer_height"]
            continuous_out[i,9] = stack["angle"]

        del inp_batches[idx]


        yield (x, [discrete_out, continuous_out])

def forward_batch_generator(batch_dir):
    gen = batch_generator(batch_dir)
    while True:
        x, y = gen.__next__()
        yield y, x

def combined_batch_validator(batch_dir):
    gen = batch_generator(batch_dir)
    while True:
        x, y = gen.__next__()
        yield y, y


def combined_batch_generator():

    while True:
        with open(args["params"], "rb") as f:
            param_dict = pickle.load(f)

        with sqlite3.connect(database=args['database']) as conn:
            crawler = Crawler(
            directory=args['s_mats'],
            cursor=conn.cursor()
            )

        spec = np.zeros((BATCH_SIZE, NUMBER_OF_WAVLENGTHS, 2))

        for i in range(BATCH_SIZE):
            #generate stacks until one doesn't block all incomming light
            while True:
                spec_x, spec_y, p1, p2, p_stack = create_random_stack(crawler, param_dict)

                if np.max(spec_x) > 0.25 or np.max(spec_y) > 0.25:
                    break

            #save the input spectrum
            spec[i] = np.stack((spec_x, spec_y), axis=1)

        yield spec, spec

#%%
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("model", metavar="m",
    	help="path to output model")
    ap.add_argument("batches", metavar="b",
    	help="path to directory containing the training batches")
    ap.add_argument("-p", "--params", default="data/smats/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-s", "--s-mats", default="data/smats",
    	help="path to the directory containing the smats")
    ap.add_argument("-log", "--log-dir", default="data/logs",
    	help="path to dir where the logs are saved")
    ap.add_argument("-n", "--new", action="store_true",
    	help="train a new model")
    ap.add_argument("-mt", "--model-type", default="inverse",
        help='["inverse", "forward", "combined"] which kind of model to train')
    ap.add_argument("-f", "--forward-model", default="data/models/best_forward.h5",
        help='needs to be provided when training a combined model')
    ap.add_argument("-i", "--inverse-model", default="data/models/best_inverse.h5",
        help='needs to be provided when training a combined model')
    ap.add_argument("-db", "--database", default="data/NN_smats.db",
                        help="sqlite database containing the adresses")
    args = vars(ap.parse_args())
    print(args)

### SETUP ###
    if args["model_type"] == "inverse":
        print("[INFO] training inverse model...")
        #continuous_out_loss = tf.Variable(1/40000)
        #callbacks = [LossWeightsChanger(continuous_out_loss)]

        if args["new"]:
            model = create_inverse_model()
            opt = Adam(lr=INIT_LR) #decay=INIT_LR / EPOCHS lr=INIT_LR,
            losses = {
                'discrete_out' : 'binary_crossentropy',
                'continuous_out' : 'mse',
                }
            metrics = {
                'discrete_out' : 'accuracy',
                'continuous_out' : 'mae',
                }
            model.compile(optimizer=opt, loss=losses, metrics=metrics)
        else:
            #the scope is nessecary beacuse I used a custom loss for training
            #with CustomObjectScope({'loss': mse_with_changable_weight(continuous_out_loss)}):
            model = load_model(args["model"])
            #Set the training generator
        generator = batch_generator

    elif args["model_type"] == "forward":

        print("[INFO] training forward model...")

        if args["new"]:
            model = create_forward_model()
            opt = Adam(lr=INIT_LR)
            model.compile(optimizer=opt, loss="mse", metrics=['mae'])
        else:
            with CustomObjectScope({'avg_init': avg_init}):
                model = load_model(args["model"])
        #Set the training generator
        generator = forward_batch_generator

    elif args['model_type'] == "combined":
        print("[INFO] training combined model...")
        if args['new']:
            #load the forward model
            with CustomObjectScope({'avg_init': avg_init}):
                try:
                    forward_model = load_model(args['forward_model'])
                except:
                    raise RuntimeError(
                        "Provide a forward model with -f when training in combined mode")


            #load the inverse model
            inverse_model = load_model(args["inverse_model"])

            #define the combined model
            forward_model.trainable = False
            x = inverse_model(forward_model.output)
            model = Model(inputs=forward_model.input, outputs=x)
            opt = Adam(lr=INIT_LR)
            losses = {
                'model' : 'binary_crossentropy',
                'model_1' : 'mse',
                }
            metrics = {
                'model' : 'accuracy',
                'model_1' : 'mae',
                }
            model.compile(optimizer=opt, loss=losses, metrics=metrics)
        else:
            with CustomObjectScope({'avg_init': avg_init}):
                model = load_model(args['model'])
        #Set the training generator
        generator = forward_batch_generator

### TRAINING ###
    trainGen = generator(f"{args['batches']}/training")
    validationGen = generator(f"{args['batches']}/validation")
    batch_count = len(os.listdir(f"{args['batches']}/training/X"))
    validation_count = len(os.listdir(f"{args['batches']}/validation/X"))

    if args['model_type'] in ['forward', 'inverse']:
        H = model.fit(
            trainGen,
    	    steps_per_epoch=batch_count,
            validation_data=validationGen,
            validation_steps=validation_count,
            epochs=EPOCHS,
            )

    elif args['model_type'] == 'combined':
        #batch_count = 400

        trainGen_combined = combined_batch_validator(
            f"{args['batches']}/training")
        validationGen_combined = combined_batch_validator(
            f"{args['batches']}/validation")



        H = model.fit(
            trainGen_combined,
            steps_per_epoch=batch_count,
            validation_data=validationGen_combined,
            validation_steps=validation_count,
            epochs=EPOCHS,
            )
        """
        for i in range(EPOCHS):
            print(f"Epoch {i+1}/{EPOCHS}")


            print("[INFO] training combined")
            forward_model.trainable = False
            combined_model.compile(optimizer=combined_opt, loss="mse", metrics=['mae'])
            #for layer in combined_model.layers[-1].layers:
                #print(layer.name, layer.trainable)
            h1 = combined_model.fit(
                trainGen_combined,
        	    steps_per_epoch=batch_count,
                validation_data=validationGen_combined,
                validation_steps=validation_count,
                epochs=1,
                )

            print("[INFO] training forward")
            forward_model = combined_model.layers[-1]
            forward_model.trainable = True
            forward_model.compile(optimizer=forward_opt, loss="mse", metrics=['mae'])
            #for layer in combined_model.layers[-1].layers:
                #print(layer.name, layer.trainable)
            #print(forward_model.summary())
            h2 = forward_model.fit(
                trainGen,
        	    steps_per_epoch=batch_count,
                validation_data=validationGen,
                validation_steps=validation_count,
                epochs=1,
                )
            H.append((h1,h2))
        """
        #model = load_inverse_from_combined(combined_model)

        #model = combined_model



    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    print("[INFO] saving logs")
    model_name = args["model"].split("/")[-1][:-3]
    with open(f"{args['log_dir']}/{model_name}.pickle", "wb") as f:
        pickle.dump(H.history, f)
    print("[DONE]")
