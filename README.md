# NeuralUTCI

A neural network emulator for the Universal Thermal Climate Index (UTCI), as described in Pastine et al.

Model is trained on 'Grid data' from Brode et al., 2012 and can be downloaded:
https://link.springer.com/article/10.1007/s00484-011-0454-1

## Installation

pip install git+https://github.com/bikempastine/NeuralUTCI.git


## Usage
```python
from NeuralUTCI import utci

result = utci(Ta=20, Tr=25, va=1.0, rH=50)
```

### Parameters

| Parameter | Description | Units | Valid Range |
|---|---|---|---|
| `Ta` | Air temperature | °C | -50 to 50 |
| `Tr` | Mean radiant temperature | °C | -80 to 120 |
| `va` | Wind speed at 10 m | m/s | 0.5 to 30.3 |
| `rH` | Relative humidity | % | 5 to 100 |

### Out-of-bounds handling

The NN is trained on the same input domain as the polynomial approximation in Bröde et al. (2012).
Inputs outside this domain are handled with the `oob` parameter:

- `oob="nan"` *(default)* — any row with one or more out-of-bounds inputs returns `NaN`
- `oob="clamp"` — inputs are clamped to the valid range before prediction, matching the behaviour described in Bröde et al. (2012)
```python
# Returns NaN for out-of-bounds inputs (default)
result = utci(Ta=20, Tr=25, va=1.0, rH=50, oob="nan")

# Clamps out-of-bounds inputs to valid range before predicting
result = utci(Ta=20, Tr=25, va=1.0, rH=50, oob="clamp")
```

## Reference

Pastine et al. ... (https://www.overleaf.com/project/6914bd3d6902c37b5585f6b3)

Bröde, P., Fiala, D., Bla˙zejczyk, K. et al. Deriving the operational procedure
    for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494
    (2012).
