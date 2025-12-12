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

IF YOU JUST WANT TO TEST WITHOUT NEED OF ACTUAL PREDICTIONS, PERFORM FOLLOWING STEPS:

1. Go to `ml_core/processing/preprocessing.py`
2. Locate `generate_time_series()` function
3. Inside the sentinel's item collection add additional argument `max_items=4`:
```python
    items = stac_client.search(
        collections=config.COLLECTION,
        intersects=polygon,
        datetime=config.DATE_RANGE,
        query={
            "platform": "sentinel-2b",
            "eo:cloud_cover": {"lt": 20}
        },
        max_items=4
    ).item_collection()
```
This will download only four sentinel-2 images for each polygon. While results may be incorrect, it may serve as a quick test of the system.
