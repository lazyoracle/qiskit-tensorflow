"""Module for testing qtf_backend
"""

from qiskit_tensorflow import QTFProvider
from qiskit_tensorflow.qtf_job import QTFJob
from qiskit.quantum_info import Statevector
from qiskit import transpile
from test.conftest import get_test_circuit  # noqa

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["qtf_qasm_simulator"])
def test_transpile(get_test_circuit, backend):  # noqa
    """Test the transpiling using our backends

    Parameters
    ----------
    get_test_circuit : callable
        pytest fixture for a simple quantum circuit
    backend : str
        name of the backend which is to be tested
    """
    qtf = QTFProvider()
    received_backend = qtf.get_backend(backend)
    trans_qc = transpile(get_test_circuit, received_backend)
    assert Statevector.from_instruction(get_test_circuit).equiv(
        Statevector.from_instruction(trans_qc)
    )


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["qtf_qasm_simulator"])
def test_run_job(get_test_circuit, backend):  # noqa
    """Test if Backend.run() gives a Job instance

    Parameters
    ----------
    get_test_circuit : callable
        pytest fixture for a simple quantum circuit
    backend : str
        name of the backend which is to be tested
    """
    qtf = QTFProvider()
    received_backend = qtf.get_backend(backend)
    trans_qc = transpile(get_test_circuit, received_backend)
    job = received_backend.run(trans_qc)
    assert isinstance(job, QTFJob)
