from qiskit.providers import ProviderV1 as Provider
from .qtf_backend import QTFQasmSimulator as QasmSimulator  # type: ignore


class QTFProvider(Provider):
    def __init__(self, token=None):
        super().__init__()
        self.name = "qtf_provider"
        self.backends = [QasmSimulator(provider=self)]

    def backends(self, name=None, **kwargs):
        if name:
            backends = [backend for backend in self.backends if backend.name() == name]
            return backends
