from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from hyperparameters import *
#This file contains formulas and other utilities that are needed by multiple
#other modules

def height_bound(periode, wav_len):
    """
    Calculates the minimum spacer height nessecary for the SASA algorithm.
    (The near field has to be sufficently decayed)

    # Arguments
        periode: int, periode of the meta surface in nm
        wav_len: float, wavelength in mu

    # Returns
        d: float, minimum spacer height
    """
    d = periode/(np.pi * np.sqrt(1 -
        periode**2 * 1.4585**2/(1e3 * wav_len)**2))
    return d/1e3

def n_SiO2_formular(w):
    """
    Calculates the refractiv index of SiO2

    Parameters
    ----------
    w : vec
        wavelengths in micro meters

    Returns
    -------
    n : vec
        refractiv indeces
    """
    a1 = 0.6961663
    a2 = 0.4079426
    a3 = 0.8974794
    c1 = 0.0684043
    c2 = 0.1162414
    c3 = 9.896161
    n = np.sqrt(a1*w**2/(w**2 - c1**2) +
        a2*w**2/(w**2 - c2**2) + a3*w**2/(w**2 - c3**2) + 1)
    return n

def mean_squared_diff(current, target, bounds=None):
    """
    Calculates the mean squared diffrence between target and current smat

    # Arguments
        current: Lx4x4 array, calculated smat from SASA
        target: Lx4x4 array, target smat of optimation

    # Returns
        output: float, real error value
    """
    if bounds is None:
        lower = 0
        upper = NUMBER_OF_WAVLENGTHS
    else:
        scale = NUMBER_OF_WAVLENGTHS/(WAVLENGTH_STOP - WAVLENGTH_START)
        lower = round(scale*bounds[0][0])
        upper = round(scale*bounds[0][1])

    return np.sum(np.abs(current[lower:upper] - target[lower:upper])**2)

def LabelBinarizer():
    discrete_params = ['Au', 'Al', 'holes', 'no holes']
    mlb = MultiLabelBinarizer(classes=np.array(discrete_params, dtype=object))
    mlb.fit_transform([['Au', 'holes']])
    return mlb

class Plotter():
    """
    Vizualizing the optimization mid process
    """
    min_x = 0
    max_x = 128


    def __init__(self, ax_num=2):
        #plt.rcParams["figure.figsize"] = (8,4)
        #Set up plot
        if ax_num == 4:
            self.figure, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(1, 4)
        elif ax_num == 3:
            self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)
            self.ax3.tick_params(
                        which='both',      # both major and minor ticks are affected
                        bottom=False,
                        left=False,     # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False,
                        labelleft=False,
                        )
        else:
            self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2)
            #Autoscale on unknown axis and known lims on the other
        self.ax1.set_autoscaley_on(True)
        self.ax1.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax1.grid()
        self.wav = np.linspace(WAVLENGTH_START, WAVLENGTH_STOP, NUMBER_OF_WAVLENGTHS)

    def save(self, stp):
        self.figure.savefig("data/plots/opti_step_{}.png".format(stp))

    def write_text(self, p1, p2, p_stack, loss_val):
        text = f"""

Layer 1:
material: {p1['particle_material']}
holes: {p1['hole']}
width: {p1['width']:.0f}
length: {p1['length']:.0f}
thickness: {p1['thickness']:.0f}
periode: {p1['periode']:.0f}

Layer 2:
material: {p2['particle_material']}
holes: {p2['hole']}
width: {p2['width']:.0f}
length: {p2['length']:.0f}
thickness: {p2['thickness']:.0f}
periode: {p2['periode']:.0f}

Stack
spacer_h: {p_stack['spacer_height']:.2f}
angle: {p_stack['angle']:.0f}
loss: {loss_val:.2f}
"""
        return text

    def update(self, current_spec, target_spec, text):

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        self.ax1.set_title("X Trans.")
        self.ax1.plot(self.wav, target_spec[:,0])
        self.ax1.plot(self.wav, current_spec[:,0])

        self.ax2.set_title("Y Trans.")
        self.ax2.plot(self.wav, target_spec[:,1])
        self.ax2.plot(self.wav, current_spec[:,1])

        self.ax3.set_title("Prediction")
        self.ax3.text(0.1, 0.05, text)

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def double_text(self, spec, pred_text, true_text):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax1.plot(spec)
        self.ax2.set_title("Prediction")
        self.ax2.text(0.1, 0.05, pred_text)
        self.ax3.set_title("True Parameters")
        self.ax3.text(0.1, 0.05, true_text)

    def double_spec(self, spec, pred_text, true_text):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax1.set_title("X Trans.")
        self.ax1.plot(spec[:,0])
        self.ax2.set_title("Y Trans.")
        self.ax2.plot(spec[:,1])
        self.ax3.set_title("Prediction")
        self.ax3.text(0.1, 0.05, pred_text)
        self.ax4.set_title("True Parameters")
        self.ax4.text(0.1, 0.05, true_text)
