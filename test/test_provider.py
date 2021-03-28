"""Module for testing QTFProvider
"""

from qiskit_tensorflow import QTFProvider
from qiskit.providers import BackendV1 as Backend

import pytest


@pytest.mark.unit
def test_backends():
    """Test backends() function which returns all available backends"""
    qtf = QTFProvider()
    for backend in qtf.backends():
        assert isinstance(backend, Backend)


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["qtf_qasm_simulator"])
def test_get_backend(backend):
    """Test get_backend() which returns the backend with matching name

    Parameters
    ----------
    backend : str
        name of the backend that is to be fetched
    """
    qtf = QTFProvider()
    received_backend = qtf.get_backend(backend)
    assert isinstance(received_backend, Backend)
