from numbers import Number
from typing import Any, Dict, Optional, Union, no_type_check

import tensorflow as tf
import numpy as np



@no_type_check
def to_numpy(x: Any) -> Union[tf.Tensor, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, tf.Tensor):  # most often case
        if tf.executing_eagerly():
            return x.numpy()
        return x.eval()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    else:  # fallback
        return np.asanyarray(x)
