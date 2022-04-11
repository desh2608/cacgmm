# GPU-based CACGMM trainer in CuPy

This package adapts the complex angular GMM model and trainer from [pb_bss](https://github.com/fgnt/pb_bss) 
using [CuPy](https://github.com/cupy/cupy) for accelerated inference on the GPU.

At the moment, it is meant to be used with the [GSS](https://github.com/desh2608/gss) toolkit, but
it can also be used as a general CACGMM trainer tool.

## Installation

```bash
> pip install cupy-cuda102  # modify according to your CUDA version (https://docs.cupy.dev/en/stable/install.html#installing-cupy)
> pip install cacgmm-gpu
```

## Usage

```python
from cacgmm.cacgmm_trainer import CACGMMTrainer

import cupy as cp

source_activity = cp.random.rand(2, 1000)
source_activity = source_activity / cp.sum(initialization, keepdims=True, axis=0)

initialization = cp.repeat(source_activity[None, ...], 513, axis=0)  # F x K x T
source_active_mask = cp.repeat(source_activity[None, ...], 513, axis=0)
X = cp.random.rand(4, 1000, 513)    # D x T x F

cacGMM = CACGMMTrainer()

cur = cacGMM.fit(
    y=X.T,
    initialization=initialization,
    iterations=10,
    source_activity_mask=source_active_mask,
)

affiliation = cur.predict(X.T, source_activity_mask=source_active_mask) # 
posterior = affiliation.transpose(1, 2, 0)  # K x T x F
```
