## Fit
The fit script ....

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

### classify_output


```python
sasa_stacker.fit.classify_output(discrete_out, continuous_out, lb)
```

