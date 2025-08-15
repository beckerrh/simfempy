# backend.py
import os
backend = os.environ.get("FEM_BACKEND", "numpy")

if backend == "jax":
    import jax
    import jax.numpy as xp
else:
    import numpy as xp
