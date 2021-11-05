"""Modeling functions and object"""

from typing import Tuple, Union
from numpy import ndarray
import numpy as np

# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes,

ArrayOrFloat = Union[ ndarray, float ]
Filters = Union[ Tuple[int,int], ndarray ]


def _smooth_abs(v: ArrayOrFloat, gamma: float = 1e-4) -> ArrayOrFloat:
    """Apply a smooth absolute value function"""
    return np.sqrt(
        gamma + v**2
    )

def _segment(X: ndarray, length: int) -> ndarray:
    """Create array of subsequences of given length and stride one"""
    return np.lib.stride_tricks.as_strided(
        x=X.squeeze(),
        shape=(X.shape[-1]-length+1, length),
        strides=(X.strides[-1], X.strides[-1]),
    )

def _sigmoid(s: ArrayOrFloat, alpha: float = 1., beta: float = 0.) -> ArrayOrFloat:
    """Apply the sigmoid function"""
    return 1 / (
        1 + np.exp((beta-s)/alpha)
    )

def _batch_norm(batch: ndarray, epsilon: float = 1e-4) -> ndarray:
    """Perform batch normalization"""
    return (
        (batch-batch.mean())
        / np.sqrt(batch.var()+epsilon)
    )


def regress(signal: ndarray, weights: ndarray, filters: ndarray,
            av: bool = True, pr: bool = True, bn: bool = False) -> ndarray:
    """Perform regression of filtering generalized additive model.

    Args:
        signal: [array] Signal array to predict on.
        weights: [array] 1D vector of weight terms. Shape = (n_filters+1, )
        filters: [array] 2D matrix of filters. Shape = (n_filters, filter_len).
        av: [bool] Whether to apply the smooth absolute value function after filtering.
        pr: [bool] Whether to apply sigmoid function before returning outputs.
        bn: [bool] Whether to apply batch normalization after aggregating.

    Returns:
        [array] Prediction array.

    Raises:
        None.
    """

    # Segment input signal
    matrix = _segment(signal, filters.shape[1])

    # Apply filters
    matrix = matrix.dot(filters.T)

    # Take absolute value
    if av:
        matrix = _smooth_abs(matrix)

    # Concatenate bias column
    matrix = np.c_[np.ones(matrix.shape[0]), matrix]

    # Apply weights
    matrix = matrix.dot(weights)

    # Batch normalize
    if bn:
        matrix = _batch_norm(matrix)

    # Calculate probability
    if pr:
        matrix = _sigmoid(matrix)

    return matrix


class Model:
    """Filtering Generalized Additive Model predictor.

    Args:
        filters: [array|tuple] 2D matrix of filters where shape = (n_filters, filter_len). Or
                 tuple of (n_filters, filter_len) for specifying shape of random initialization.
        weights: [ndarray] 1D vector of weight terms where shape = (n_filters+1, ). Default to -1
                 to match shape of filters matrix for random initialization.
        av: [bool] Whether to apply the smooth absolute value function after filtering.
        pr: [bool] Whether to apply sigmoid function before returning outputs.
        bn: [bool] Whether to apply batch normalization after aggregating.
        seed: [int] Seed for random initialization.

    Raises:
        None.
    """

    def __init__(self, *,
        filters: Filters = (5,5),
        weights: ndarray = -1,
        av: bool = True,
        pr: bool = True,
        bn: bool = False,
        seed: bool = None):

        self.rng = np.random.default_rng(seed)

        self.filters = filters
        self.weights = weights

        self.av = av
        self.pr = pr
        self.bn = bn

    def __call__(self, *args, **kwargs):
        """Wrapper around prediction method"""

        return self.predict(*args, **kwargs)

    @property
    def filters(self) -> ndarray:
        """Filters matrix, each row is an individual filter"""

        return self._filters

    @filters.setter
    def filters(self, filters_in: Filters):
        """Filters matrix setter. Accepts 2D array or length-2 shape tuple"""

        if isinstance(filters_in, np.ndarray) and len(filters_in.shape)==2:
            self._filters = filters_in.copy()
        elif isinstance(filters_in, tuple) and len(filters_in)==2:
            self._filters = self.rng.normal(size=np.product(filters_in)).reshape(filters_in)
        else:
            raise TypeError(f'filters input must be 2D array or length-2 tuple. Not {filters_in}')

    @property
    def weights(self) -> ndarray:
        """Weight vector, length = n_filters + 1"""

        return self._weights

    @weights.setter
    def weights(self, weights_in: ndarray):
        """Weight vector setter. Accepts 1D array or -1"""

        if isinstance(weights_in, np.ndarray) and len(weights_in.shape)==1:
            self._weights = weights_in.copy()
        elif isinstance(weights_in, int) and  weights_in==-1:
            self._weights = self.rng.normal(size=self.filters.shape[0]+1)
        else:
            raise TypeError(f'weights input must be 1D array or -1. Not {weights_in}')

    @staticmethod
    def _validate(signal: ndarray, weights: ndarray, filters: ndarray):
        """Validate input arrays"""

        if not len(weights.shape) == 1:
            raise ValueError(f'Weights vector must be 1-dimmensional. Not shape {weights.shape}')
        if not len(signal.shape) == 1:
            raise ValueError(f'Signal must be 1-dimmensional. Not shape {signal.shape}')
        if not len(filters.shape) == 2:
            raise ValueError(f'Filters matrix must be 2-dimmensional. Not shape {weights.shape}')
        if not weights.shape[0] == filters.shape[0]+1:
            raise ValueError('Weights vector length must equal number of filters (first dim) - 1.'
                             f' Not weights {weights.shape} and filters {filters.shape}')

    def predict(self, signal: ndarray,
                weights: ndarray = None,
                filters: ndarray = None,
                fill: float = None) -> ndarray:
        """Make prediction on input data with given weights and filters.

        Args:
            signal: [array] Signal array to predict on.
            weights: [array] Optional 1D vector of weight terms. Shape = (n_filters+1, )
            filters: [array] Optional 2D matrix of filters. Shape = (n_filters, filter_len).
            fill: [float] Value to fill/pad prediction with to match input signal length.

        Returns:
            [array] Prediction array.

        Raises:
            ValueError if weights is not 1-dimmensional.
            ValueError if signal is not 1-dimmensional.
            ValueError if filters is not 2-dimmensional.
            ValueError if n_filters+1 does not equal weights_len.
        """

        signal = np.asarray(signal).squeeze()

        if weights is None:
            weights = self.weights.copy()
        if filters is None:
            filters = self.filters.copy()

        self._validate(signal, weights, filters)

        pred = regress(
            signal=signal,
            weights=weights,
            filters=filters,
            av=self.av,
            pr=self.pr,
            bn=self.bn,
        )

        if fill:
            offset = (signal.shape[0]-pred.shape[0])
            pred = np.hstack([
                np.full(offset//2, fill),
                pred.squeeze(),
                np.full(offset//2, fill),
            ])

        return pred
