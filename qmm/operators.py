# Q-MM: A Python Quadratic Majorization Minimization toolbox
# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Implements high level interface to manipulate linear operators. This module
is not required by Q-MM, is basic, but can serve as guide or for reuse.

"""

import abc
from typing import List, Sequence, Tuple, Union

import numpy as np  # type: ignore
from numpy import ndarray as array

ArrOrList = Union[array, List[array]]


def dft2(obj: array) -> array:
    """Return the orthogonal real 2D fft

    Parameters
    ----------
    obj : array-like
        The array on which to perform the 2D DFT

    Returns
    -------
    out : array

    Notes
    -----

    This function is a wrapper of numpy.fft.rfft2. FFT is made on 2 last axis.

    """
    return np.fft.rfft2(obj, norm="ortho")


def idft2(obj: array, shape: Tuple[int]) -> array:
    """Return the orthogonal real 2D ifft

    Parameters
    ----------
    obj : array-like
        The array on which to perform the inverse 2D DFT

    shape: tuple
        The output shape

    Returns
    -------
    out : array

    Notes
    -----

    This function is a wrapper of numpy.fft.irfft2. FFT is made on 2 last axis.

    """
    return np.fft.irfft2(obj, norm="ortho", s=shape)


class Operator(abc.ABC):
    """An abstract base class for linear operators"""

    @abc.abstractmethod
    def forward(self, point: array) -> ArrOrList:
        """Return H·x"""
        return NotImplemented

    @abc.abstractmethod
    def adjoint(self, point: ArrOrList) -> array:
        """Return Hᵗ·e"""
        return NotImplemented

    def fwback(self, point: array) -> array:
        """Return HᵗH·x"""
        return self.adjoint(self.forward(point))

    def T(self, point: ArrOrList) -> array:
        """Return Hᵗ·e"""
        return self.adjoint(point)

    def __call__(self, point: array) -> ArrOrList:
        """Return H·x"""
        return self.forward(point)


class Conv2(Operator):
    """The 2D convolution on image

    Does not suppose periodic or circular condition. Has the following
    attributes

    Attributes
    ----------
    imp_resp : array
        The impulse response.
    shape : tuple of int
        The of the input image.
    freq_resp : array
        The frequency response of shape `shape`.

    Notes
    -----
    Use the fft internally for fast computation.

    """

    def __init__(self, ir: array, shape: Tuple[int]):
        """The 2D convolution on image

        Parameters
        ----------
        ir : array
            The impulse response
        shape : tuple of int
            The of the input image

        """

        self.imp_resp = ir
        self.shape = shape
        self._ir_shape = ir.shape
        self.freq_resp = ir2fr(ir, self.shape)

    def forward(self, point: array) -> array:
        return idft2(dft2(point) * self.freq_resp, self.shape)[
            : -self._ir_shape[0] + 1, : -self._ir_shape[1] + 1
        ]

    def adjoint(self, point: array) -> array:
        out = np.zeros(self.shape)
        out[: point.shape[0], : point.shape[1]] = point
        return idft2(dft2(out) * self.freq_resp.conj(), self.shape)

    def fwback(self, point: array) -> array:
        out = idft2(dft2(point) * self.freq_resp, self.shape)
        out[-self._ir_shape[0] + 1 :, :] = 0
        out[:, -self._ir_shape[1] + 1 :] = 0
        return idft2(dft2(out) * self.freq_resp.conj(), self.shape)


class Diff(Operator):
    """The difference operator

    Compute the first-order differences along an axis.

    Attributes
    ----------
    axis : int
        The axis along which the differences is performed

    Notes
    -----
    Use `numpy.diff` and implement the correct adjoint, with `numpy.diff` also.

    """

    def __init__(self, axis):
        """The first-order differences operator

        Parameters
        ----------
        axis: int
            the axis along which to perform the diff"""
        self.axis = axis

    def response(self, ndim):
        """Return the equivalent impulse response.

        The result of `forward` method is equivalent with the 'valid'
        convolution with this response.

        The adjoint operator corresponds the 'full' convolution with flipped
        response.

        """
        imp_resp = np.zeros(ndim * [2])
        index = ndim * [0]
        index[self.axis] = slice(None, None)
        imp_resp[tuple(index)] = [1, -1]
        return imp_resp

    def freq_response(self, ndim, shape):
        """The frequency response."""
        return ir2fr(self.response(ndim), shape)

    def forward(self, point):
        return np.diff(point, axis=self.axis)

    def adjoint(self, point):
        return -np.diff(point, prepend=0, append=0, axis=self.axis)


def ir2fr(
    imp_resp: array,
    shape: Sequence[int],
    center: Sequence[int] = None,
    real: bool = True,
) -> array:
    """Return the frequency response from impulse responses.

    This function make the necessary correct zero-padding, zero convention,
    correct DFT etc. to compute the frequency response from impulse responses
    (IR).

    The IR array is supposed to have the origin in the middle of the array.

    The Fourier transform is performed on the last `len(shape)` dimensions.

    Parameters
    ----------
    imp_resp : array
        The impulse responses.
    shape : tuple of int
        A tuple of integer corresponding to the target shape of the frequency
        responses, without hermitian property. `len(shape) >= ndarray.ndim`. The
        DFT is performed on the `len(shape)` last axis of ndarray.
    center : tuple of int, optional
        The origin index of the impulse response. The middle by default.
    real : boolean, optional
        If True, imp_resp is supposed real, the hermitian property is used with
        rfftn DFT and the output has `shape[-1] / 2 + 1` elements on the last
        axis.

    Returns
    -------
    out : array
       The frequency responses of shape `shape` on the last `len(shape)`
       dimensions.

    Notes
    -----

    - The output is returned as C-contiguous array.

    - For convolution, the result have to be used with unitary discrete Fourier
      transform for the signal (norm="ortho" of fft).

    - DFT are always performed on last axis for efficiency (C-order array).

    """
    if len(shape) > imp_resp.ndim:
        raise ValueError("length of shape must be inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2)) for length in imp_resp.shape[-len(shape) :]]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(imp_resp.shape[: -len(shape)] + shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at index 0, the
    # hypothesis of the DFT
    for axe, shift in enumerate(center):
        irpadded = np.roll(irpadded, -shift, imp_resp.ndim - len(shape) + axe)

    # Perform the DFT on the last axes
    if real:
        freq_resp = np.ascontiguousarray(
            np.fft.rfftn(
                irpadded, axes=list(range(imp_resp.ndim - len(shape), imp_resp.ndim))
            )
        )
    else:
        freq_resp = np.ascontiguousarray(
            np.fft.fftn(
                irpadded, axes=list(range(imp_resp.ndim - len(shape), imp_resp.ndim))
            )
        )
    return freq_resp
