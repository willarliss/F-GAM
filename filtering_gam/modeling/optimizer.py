"""Training functions and object"""

from typing import List, Callable, Tuple
from numpy import ndarray
import numpy as np

from .model import Model

# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes,
# pylint: disable=too-many-locals,inconsistent-return-statements

WeightsFilters = Tuple[ ndarray, ndarray ]
Data = List[ndarray]


def _cross_entropy(y_true: ndarray, y_prob: ndarray, eps: float = 1e-9) -> ndarray:
    """Calculates binary cross-entropy loss"""
    return -1 * (
        np.log(y_prob+eps)*y_true + np.log(1-y_prob+eps)*(1-y_true)
    )

def _mse(y_true: ndarray, y_pred: ndarray) -> ndarray:
    """Calculates mean-squared-error loss"""
    return np.abs(
        y_true - y_pred
    ) ** 2

def _split_params(array: ndarray, shape: int) -> WeightsFilters:
    """Split vector of parameters into weights vector and filters matrix"""
    return (
        array[:shape], # Weights
        array[shape:].reshape(shape-1,-1), # Filters
    )

def _slicing(diff: int) -> Tuple[int,int]:
    """Array slicing for matching target array size to prediction array size"""
    offset = diff//2
    return offset, (None if offset==0 else -offset)

def _finite_difference(func: Callable, params: ndarray, eps: float = 1.1e-16) -> ndarray:
    """Calculate centeral finite difference to return gradient"""

    len_ = params.shape[0]
    grad = np.full(len_, np.nan)

    for idx in range(len_):
        e_i = np.zeros(len_)
        e_i[idx] = np.sqrt(eps)

        args1 = params.copy() + (e_i)
        args2 = params.copy() - (e_i)

        grad[idx] = (func(args1) + -func(args2)) / (2*e_i.sum())

    return grad


class Optimizer:
    """Filtering Generalized Additive Model optimizer.

    Args:
        model: [obj] Model to train with.
        weight: [bool] Whether to use weighted average in loss function.
        loss: [str] Type of loss function to use.
        penalty: [str] Type of penalty to use in loss function.
        epochs: [int] Number of epochs to train for.
        gamma: [float] Learning rate decay parameter.
        alpha: [float] Regularization strength for penalty in loss function.
        eta0: [float] Initial learning rate.
        tol: [float] Stopping tolerance.

    Raises:
        ValueError if gamma is not in the interval (0,1].
        ValueError if alpha is negative.
        ValueError if eta0 is not in the interval [0,1).
        ValueError if tol is not greater than 0.
    """

    def __init__(self, model: Model, *,
                 weight: bool = True,
                 loss: str = 'entropy',
                 penalty: str = None,
                 epochs: int = 20,
                 gamma: float = 0.,
                 alpha: float = 0.1,
                 eta0: float = 0.01,
                 tol: float = 0.001):

        self.model = model

        self.weight = weight
        self.loss = loss
        self.penalty = penalty
        self.epochs = epochs

        self.gamma = gamma
        if not 0. <= self.gamma < 1.:
            raise ValueError(f'gamma must be in the interval (0,1]. Not {self.gamma}')
        self.alpha = alpha
        if self.alpha < 0.:
            raise ValueError(f'alpha must be positive. Not {self.alpha}')
        self.eta0 = eta0
        if not 0. < self.eta0 <= 1.:
            raise ValueError(f'eta must be in the interval [0,1). Not {self.eta0}')
        self.tol = tol
        if self.tol <= 0.:
            raise ValueError(f'tolerance must be greater than 0. Not {self.tol}')
        if self.tol >= 1.:
            print(f'Warning: tolerance={self.tol} is high.')

    def eta_schedule(self, eta: float, step: int) -> float:
        """Learning rate decay schedule.

        Args:
            eta: [float] Current learning rate.
            step: [int] Current training step.

        Returns:
            [float] New learning rate.

        Raises:
            None.
        """

        decay = 1 / (self.gamma*step + 1)

        return eta * decay

    def define_loss_fn(self, X: ndarray, y: ndarray, shape: int) -> Callable:
        """Wrapper around loss function for defining constants.

        Args:
            X: [array] 2D array of explanatory variables.
            y: [array] 1D array of target variable.
            shape: [int] Number of weight terms.

        Returns:
            [callable] Loss function.

        Raises:
            ValueError if unknown penalty type is specified on init.
            ValueError if unknown loss function is specified on init.
        """

        if self.penalty == 'l1':
            pen = lambda x: np.linalg.norm(x, 1) * self.alpha
        if self.penalty == 'l2':
            pen = lambda x: np.linalg.norm(x, 2) * self.alpha
        if (self.penalty is None) or (self.penalty == 'none'):
            pen = lambda x: 0.
        else:
            raise ValueError(f'Unknown penalty type: {self.penalty}')

        aggregate = np.full(y.shape, 1/y.shape[0])

        if self.loss == 'entropy':

            if self.weight:
                aggregate = np.where(y==1, 1-y.mean(), y.mean())

            def loss(params):
                weights, filters = _split_params(params, shape)
                pred = self.model(X, weights, filters)
                off0, off1 = _slicing(y.shape[0]-pred.shape[0])
                entropy = _cross_entropy(y[off0:off1], pred)
                return entropy.dot(aggregate[off0:off1]) + pen(params)

        elif self.loss == 'mse':

            def loss(params):
                weights, filters = _split_params(params, shape)
                pred = self.model(X, weights, filters)
                offset = pred.shape[0] # Slicing?
                error = _mse(y[:offset], pred)
                return error.dot(aggregate[:offset]) + pen(params)

        else:
            raise ValueError(f'Unknown loss function: {self.loss}')

        return loss

    @staticmethod
    def _validate(X: ndarray, y: ndarray):
        """Validate input data"""

        message = (
            'X and y inputs must be lists/tuples containing 1-dimmensional numpy arrays.'
            ' Each item in X must be the same length as the corresponding item in y.'
        )

        if not isinstance(X, (list, tuple)):
            raise ValueError(message)
        if not isinstance(y, (list, tuple)):
            raise ValueError(message)
        if len(X) != len(y):
            raise ValueError(message)

        for idx, _ in enumerate(X):
            if not isinstance(X[idx], np.ndarray) and len(X[idx].shape)==1:
                raise ValueError(message)
            if not isinstance(y[idx], np.ndarray) and len(y[idx].shape)==1:
                raise ValueError(message)
            if X[idx].shape != y[idx].shape:
                raise ValueError(message)

    def train(self, X: Data, y: Data,
              return_params=False) -> WeightsFilters:
        """Perform gradient descent on model parameters given training data inputs.

        Args:
            X: [list(array)] List of numpy arrays of signals to train on.
            y: [list(array)] List of numpy arrays of labels to train on.
            return_params: [bool] Whether to return trained parameters or update model inplace.

        Returns:
            [array,array] Weights vector and filters matrix if specified.

        Raises:
            ValueError if X and y are not lists/tuples containing 1D numpy arrays where each
                item in X matches in length the corresponding item in y.
        """

        self._validate(X, y)

        # Initialize with weights from model
        weights0 = self.model.weights.copy()
        filters0 = self.model.filters.copy()

        shape = weights0.shape[0]
        eta = float(self.eta0)

        step = 0
        for _ in range(self.epochs):

            diff = []
            for X_i, y_i in zip(X, y):

                # Define loss function given current signal and corresponding labels
                loss = self.define_loss_fn(X_i, y_i, shape)

                # Flatten weightrs and filters into one array
                params = np.hstack([weights0, filters0.flatten()])

                # Calculate gradient with finite differencing
                gradient = _finite_difference(loss, params)
                weights_grad, filters_grad = _split_params(gradient, shape)

                # Update params
                weights1 = weights0 - eta*weights_grad
                filters1 = filters0 - eta*filters_grad

                # Store change in parameters
                diff.append(
                    np.abs(weights0-weights1).sum()
                    +np.abs(filters0-filters1).sum()
                )

                # Reassign weights, update learning rate and step count
                weights0, filters0 = weights1, filters1
                eta = self.eta_schedule(eta, step)
                step += 1

            # Stop if average difference in last epoch is less than tolerance
            if np.mean(diff) < self.tol:
                break

        if return_params:
            return weights0, filters0

        # Upate model parameters inplace
        self.model.weights = weights0.copy()
        self.model.filters = filters0.copy()
