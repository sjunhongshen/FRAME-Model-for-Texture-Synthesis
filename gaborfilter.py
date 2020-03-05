import math
import numpy as np

def gaborfilter(size, orientation):
    """
      [Cosine, Sine] = gaborfilter(scale, orientation)

      Defintion of "scale": the sigma of short-gaussian-kernel used in gabor.
      Each pixel corresponds to one unit of length.
      The size of the filter is a square of size n by n.
      where n is an odd number that is larger than scale * 6 * 2.
    """

    assert size % 2 != 0

    halfsize = math.ceil(size / 2)
    theta = (math.pi * orientation) / 180
    Cosine = np.zeros((size, size))
    Sine = np.zeros((size, size))
    gauss = np.zeros((size, size))
    scale = size / 6

    for i in range(size):
        for j in range(size):
            x = ((halfsize - (i+1)) * np.cos(theta) + (halfsize-(j+1)) * np.sin(theta)) / scale
            y = (((i+1) - halfsize) * np.sin(theta) + (halfsize-(j+1)) * np.cos(theta)) / scale

            gauss[i, j] = np.exp(-(x**2 + y**2/4) / 2)
            Cosine[i, j] = gauss[i, j] * np.cos(2*x)
            Sine[i, j] = gauss[i, j] * np.sin(2*x)

    k = np.sum(np.sum(Cosine)) / np.sum(np.sum(gauss))
    Cosine = Cosine - k * gauss

    return Cosine, Sine


def test():
    [a, b] = gaborfilter(3, 0)
    a_true = np.array([-0.1068, -0.1761, -0.1068, 0.2136, 0.3522, 0.2136, -0.1068, -0.1761, -0.1068])
    assert np.sum(np.abs(a.reshape(-1) - a_true)) < 1e-3


if __name__ == '__main__':
    test()
