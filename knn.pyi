import numpy as np
from numpy.typing import NDArray

class KDTree:
    """
    A KDTree implemented in Rust and PyO3.
    
    :param train: the input train data set to build the KDTree, must be a 2-D ndarray of i64, e.g. np.array([[1, 2], [3, 4], [5, 6]])
    """
    def __init__(self, train: NDArray[np.int64]):
        ...

    def k_nearest(self, query: NDArray[np.int64], k: int) -> NDArray[np.int64]:
        """
        Get the top K nearest vectors of the given query vector, 

        :param query: the query vector, must be a 1-D ndarray of i64, e.g. np.array([1, 2])
        :param k: the number of return vectors
        :return: a array of the nearest K vectors
        """
        ...

def knn(train: NDArray[np.int64], query: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    A shorthand for KDTree(train).k_nearest(query, k)
    """
    ...
