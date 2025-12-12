## How to make prediction? 

```python
>>> from ml_core import predict
>>> results = predict([(1232.0, '1285'), (2.0, '2')] 
>>> results
<<< {'ids': array([ 676159631400940591, 7661089566532927138]),
 'y_true': array(["Bog'", 'Paxta'], dtype='<U5'),
 'y_pred': array(['Paxta', "G'alla"], dtype='<U6')}
```

_!IMPORTANT!_

For prediction, you must pass at least two pairs of `(kontur_raqami, kesma_raqami)`