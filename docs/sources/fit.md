# Fit.py
This is the main function taking a transmission spectrum as input and producing a metasurface stack as output.

## Usage
Help to all scripts can be revived with the `-h` option. `fit -h`:

<pre><code>
fit.py [-h] [-m MODEL] [-db DATABASE] [-S SMATS] [-i INDEX] [-I] s

positional arguments:
  s                     path to target spectrum .npy file

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to trained model model
  -db DATABASE, --database DATABASE

                        sqlite database containing the adresses
  -S SMATS, --smats SMATS
                        directory containing the smats for interpolation
  -i INDEX, --index INDEX
  -I, --interpolate
 </code></pre>

The target spectrum has to be provided as a `.npy` array of shape `L x 2` where `L` is the number of sampled wavelengths and the `2` contains X - and Y - polarization. The provided model `stacker.h5` has been trained on a dataset with `L = 160`

___

## Source Code



<span style="float:right;">[[source]](https://github.com/TimLucaTuran/stacker/tree/master/sasa_stacker/fit.py#L19)</span>
### SingleLayerInterpolator

```python
sasa_stacker.fit.SingleLayerInterpolator(crawler, num_of_neigbours=6, power_faktor=2)
```


This class takes parameters of a single layer meta surface and
looks into the database for similar layers which have been simulated. It then
interpolates these to get an approximation for the behaviour of a layer
with the provided parameters.

__Arguments__

- __crawler__: crawler obj.
- __num_of_neigbours__: int, how many similar layers should be
    considered for the interpolation
- __power_faktor__: int, exponent for inverse-distance-weight interpolation (IDW)



----

### loss


```python
sasa_stacker.fit.loss(arr, target_spec, p1, p2, p_stack, b1, b2, b_stack, crawler, plotter, sli, stp)
```



This loss function is minimized by the scipy optimizer. It takes all the
parameters of a stack, calculates the resulting transmission spectrum and
compares it to the target. Additionally it checks if physical bounds are
violated and adds `params_bounds_distance()` to the loss value.

__Arguments__

- __arr__: array, the scipy optimizer needs the first argument to be an array
    with all the tuneable parameters.
- __target_spec__: Lx2 array
- __p1__: dict, parameters of layer 1
- __p2__: dict, parameters of layer 2
- __p_stack__: dict, parameters of the stack
- __bounds__: dict, {parameter: [lower bound, upper bound]}
- __crawler__: crawler object to access the db
- __plotter__: plotter object
- __sli__: SingleLayerInterpolator object

__Returns__

- __loss_val__: float


