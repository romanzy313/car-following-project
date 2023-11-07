import pytest

import pandas as pd

from src.utils import create_sequences_2
import logging

# LOGGER = logging.getLogger(__name__)


class TestUtils:
    def test_sequence(self):
        raw_data = pd.DataFrame(
            {
                "delta_pos": [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                "delta_vel": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            }
        )

        (X, y) = create_sequences_2(raw_data, 3, 2, 2)

        print("X is ", X)
        print("y is", y)

        assert False

        pass
