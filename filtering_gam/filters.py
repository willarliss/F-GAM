"""Filter generator class"""

from typing import List, Tuple
from numpy import ndarray
import numpy as np

FiltersAndArgs = List[Tuple[str,dict]]

class Filters:
    """Generate filters for 1-dimmensional convolutions.

    Args:
        length: [int] Length of the filters to generate.
        random_state: [int] Random state for random number generator.

    Raises:
        None.
    """

    def __init__(self, length: int, *,
                 random_state: int = None):

        self.rng = np.random.default_rng(random_state)

        self.length = length
        self.start = -int(np.floor(length/2))
        self.center = length//2
        self.end = int(np.ceil(length/2))

    @staticmethod
    def mmscale(array: ndarray, max_: float = 1., min_: float = -1.) -> ndarray:
        """Apply Min-Max scaling to an array.

        Args:
            array: [array] Array to be scaled.
            max_: [float] Maximum value of output array.
            min_: [float] Minimum value of output array.

        Returns:
            [array] Scaled array.

        Raises:
            None.
        """

        scaled = (array-array.min()) / (array.max()-array.min())

        return scaled * (max_-min_) + min_

    def multiple(self, filters: FiltersAndArgs) -> ndarray:
        """Return matrix of multiple filters as specified by inputs.
        e.g. filters = [('sin', {'pos': True}),]

        Args:
            filters: [list] List of tuples. Each tuple contains a string corresponding to a
                filter name and a dictionary corresponding to filter key-word arguments.

        Returns:
            [array] A 2D matrix where each row is a filter.

        Raises:
            None.
        """

        fltrs_matrix = []
        for fltr in filters:
            fltrs_matrix.append(
                getattr(self, fltr[0])(**fltr[1])
            )

        return np.array(fltrs_matrix)

    def sin(self, pos: bool = True) -> ndarray:
        """Sine wave function shaped filter"""
        line = np.linspace(0., 2., self.length)
        line = np.sin(line*np.pi) * (1. if pos else -1.)
        return self.mmscale(line)

    def cos(self, pos: bool = True) -> ndarray:
        """Cosine wave function shaped filter"""
        line = np.linspace(0., 2., self.length)
        line = np.cos(line*np.pi) * (1. if pos else -1.)
        return self.mmscale(line)

    def tan(self, pos: bool = True) -> ndarray:
        """Tangent function shaped filter"""
        line = np.linspace(self.start, self.end, self.length)
        line = np.tan(line*np.pi) * (1. if pos else -1.)
        return self.mmscale(line)

    def tanh(self, alpha: float = 1., beta: float = 0.) -> ndarray:
        """Hyperbolic tangent function shaped filter"""
        line = np.linspace(self.start, self.end, self.length)
        line = np.tanh((line-beta)/alpha)
        return self.mmscale(line)

    def sinh(self, alpha: float = 1., beta: float = 0.) -> ndarray:
        """Hyperbolic sine wave function shaped filter"""
        line = np.linspace(self.start, self.end, self.length)
        line = np.sinh((line-beta)/alpha)
        return self.mmscale(line)

    def identity(self) -> ndarray:
        """Identity filter"""
        coefs = np.zeros(self.length)
        coefs[self.center] = 1.
        return coefs

    def shift(self, offset: int = 0) -> ndarray:
        """Identity-shift filter"""
        coefs = np.zeros(self.length)
        coefs[(self.center)+offset] = -1.
        return coefs

    def mean(self) -> ndarray:
        """Moving average filter"""
        coefs = np.full(self.length, 1/self.length)
        return coefs

    def full(self) -> ndarray:
        """Moving sum filter"""
        coefs = np.ones(self.length)
        return coefs

    def matching(self, pos: bool = True) -> ndarray:
        """Matching filter"""
        coefs = np.ones(self.length) * -1
        coefs[self.center:] *= -1
        return coefs * (1. if pos else -1.)

    def derivative(self, pos: bool = True) -> ndarray:
        """Derivative filter"""
        line = np.linspace(-1., 1., self.length)
        return line * (1. if pos else -1.)

    def gaussian(self, loc: float = 0., scale: float = 1.) -> ndarray:
        """Gaussian noise filter"""
        noise = self.rng.normal(loc=loc, scale=scale, size=self.length)
        return self.mmscale(noise)

    def tesla(self, pos: bool = True) -> ndarray:
        """Tesla-T shaped filter"""
        coefs = np.hstack([
            np.linspace(0., 1/3, self.center),
            np.array(-2/3),
            np.linspace(1/3, 0., self.length-self.center-1)
        ])
        return coefs * (1. if pos else -1.)

    def low_pass(self, *args, **kwargs):
        """Low-pass filter"""
        raise NotImplementedError

    def high_pass(self, *args, **kwargs):
        """High-pass filter"""
        raise NotImplementedError
