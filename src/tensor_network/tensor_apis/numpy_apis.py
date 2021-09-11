from tensor_network.tensor_apis.base_api import BaseTensorAPI, OutOfMemoryError


class NumpyAPI(BaseTensorAPI):
    def __init__(self):
        import numpy

        self._thread_limit = None
        self._thread_limiter = None
        self._numpy = numpy
        self._entry_type = self._numpy.float64

    def add_argument(self, key, value):
        if key == "entry_type":
            types = {
                "float64": self._numpy.float64,
                "float32": self._numpy.float32,
                "float16": self._numpy.float16,
                "uint": self._numpy.uint64,
                "int": self._numpy.int64,
                "bigint": self._numpy.object,
            }

            if value in types:
                self._entry_type = types[value]
            else:
                raise ValueError("Unknown numpy type %s" % value)
        elif key == "thread_limit":
            import threadpoolctl

            self._thread_limit = value
            self._thread_limiter = threadpoolctl
        else:
            super(NumpyAPI, self).add_argument(key, value)

    def create_tensor(self, shape, default_value=None):
        if default_value is None:
            return self._numpy.empty(shape, dtype=self._entry_type)
        else:
            return self._numpy.full(shape, default_value, dtype=self._entry_type)

    def tensordot(self, a, b, axes):
        return self._numpy.tensordot(a, b, axes)

    def contract(self, network, contraction_tree):
        try:
            if self._thread_limit is not None:
                with self._thread_limiter.threadpool_limits(
                    limits=self._thread_limit, user_api="blas"
                ):
                    return network.identify(contraction_tree, self)
            else:
                return network.identify(contraction_tree, self)
        except MemoryError:
            raise OutOfMemoryError

    def warm(self):
        """
        Warm up the library, if needed.
        :return: None
        """
        pass

    def get_entry_size(self):
        return self._numpy.dtype(self._entry_type).itemsize


class NumpyLogAPI(NumpyAPI):
    def __init__(self):
        super().__init__()
        self.lognorm = 0

    def get_value(self, tensor, tup):
        return tensor.lognorm + self._numpy.log(tensor[tup])

    def create_tensor(self, shape, default_value=None):
        if default_value is None:
            return NumpyTensorLog(0, self._numpy.empty(shape, dtype=self._entry_type))
        else:
            return NumpyTensorLog(0, self._numpy.full(shape, default_value, dtype=self._entry_type))

    def tensordot(self, a, b, axes):
        c = NumpyTensorLog(0, self._numpy.tensordot(a.nparray, b.nparray, axes))
        self.normalize(c)
        c.lognorm += a.lognorm + b.lognorm
        return c

    def normalize(self, a):
        norm = self._numpy.amax(a.nparray)
        if norm !=0:
            a.nparray /= norm
            a.lognorm += self._numpy.log(norm)

class NumpyTensorLog():
    def __init__(self, lognorm, nparray):
        self.lognorm = lognorm
        self.nparray = nparray

    def __getitem__(self, index):
        return self.nparray[index]

    def __setitem__(self, key, value):
        self.nparray[key] = value



NUMPY_APIS = {
    "numpy": NumpyAPI,
    "numpylog": NumpyLogAPI
}
