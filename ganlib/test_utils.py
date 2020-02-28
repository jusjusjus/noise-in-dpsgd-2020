
import numpy as np
import pytest

from .utils import iterate_batched


@pytest.mark.parametrize("batch_size", [None, 1, 3, 4])
def test_iterate_batched(batch_size):
    x = np.arange(20)
    y = np.sort(np.concatenate(list(iterate_batched(x, batch_size))))
    assert np.all(x == y)
