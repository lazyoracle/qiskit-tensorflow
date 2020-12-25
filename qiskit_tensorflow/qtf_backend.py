import warnings

from qiskit import qobj
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.providers.models import BackendConfiguration
from qiskit.util import deprecate_arguments
from qiskit.result import Result

from .qtf_job import QTFJob

from typing import Union


class QTFQasmSimulator(Backend):
    MAX_QUBITS_MEMORY = 10
    DEFAULT_CONFIGURATION = {
        "backend_name": "qtf_qasm_simulator",
        "backend_version": "0.0.1",
        "n_qubits": min(100, MAX_QUBITS_MEMORY),
        "url": "https://github.com/lazyoracle/qiskit-tensorflow",
        "simulator": True,
        "local": True,
        "conditional": True,
        "open_pulse": False,
        "memory": True,
        "max_shots": 65536,
        "coupling_map": None,
        "description": "A tensorflow simulator for qasm experiments",
        "basis_gates": ["u1", "u2", "u3", "cx", "id", "unitary"],
        "gates": [
            {
                "name": "u1",
                "parameters": ["lambda"],
                "qasm_def": "gate u1(lambda) q { U(0,0,lambda) q; }",
            },
            {
                "name": "u2",
                "parameters": ["phi", "lambda"],
                "qasm_def": "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }",
            },
            {
                "name": "u3",
                "parameters": ["theta", "phi", "lambda"],
                "qasm_def": "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }",
            },
            {
                "name": "cx",
                "parameters": ["c", "t"],
                "qasm_def": "gate cx c,t { CX c,t; }",
            },
            {
                "name": "id",
                "parameters": ["a"],
                "qasm_def": "gate id a { U(0,0,0) a; }",
            },
            {
                "name": "unitary",
                "parameters": ["matrix"],
                "qasm_def": "unitary(matrix) q1, q2,...",
            },
        ],
    }

    SHOW_FINAL_STATE = False

    def __init__(self, configuration=None, provider=None):
        super().__init__(
            configuration=(
                configuration
                or BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)
            ),
            provider=provider,
        )

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024, memory=False, initial_statevector=None, chop_threshold=1e-15
        )

    @deprecate_arguments({"qobj": "circuit"})
    def run(self, circuit: Union[qobj.Qobj, QuantumCircuit], **kwargs) -> QTFJob:
        """Parse and run a Qobj or QuantumCircuit. Sync only

        Parameters
        ----------
        circuit : [Qobj, QuantumCircuit]
            The Qobj or QuantumCircuit to be simulated

        Returns
        -------
        QTFJob
            An instance of the QTFJob with the result

        Raises
        ------
        ValueError
            When number of shots is greater than max_shots
        QiskitError
            Support for Pulse Jobs is not implemented
        """
        if isinstance(circuit, qobj.QasmQobj):
            warnings.warn(
                "Passing in a QASMQobj object to run() is "
                "deprecated and will be removed in a future "
                "release",
                DeprecationWarning,
            )
            if circuit.config.shots > self.configuration().max_shots:
                raise ValueError(
                    "Number of shots is larger than maximum " "number of shots"
                )
            # TODO parse qobj to json
        elif isinstance(circuit, qobj.PulseQobj):
            raise QiskitError("Pulse jobs are not accepted")
        else:
            for kwarg in kwargs:
                if kwarg != "shots":
                    warnings.warn(
                        "Option %s is not used by this backend" % kwarg,
                        UserWarning,
                        stacklevel=2,
                    )
            out_shots = kwargs.get("shots", self.options.shots)
            if out_shots > self.configuration().max_shots:
                raise ValueError(
                    "Number of shots is larger than maximum " "number of shots"
                )
            # TODO parse circuit to json

        # TODO implement internal methods for simulating circuit or qobj

        res = Result()
        job = QTFJob(self, None, res)
        return job
