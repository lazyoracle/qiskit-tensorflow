import uuid
import time
import numpy as np
import logging

from qiskit.util import local_hardware_info
from qiskit import qobj
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.compiler import assemble
from qiskit.qobj.qasm_qobj import QasmQobjExperiment

from .qtf_exceptions import QTFError
from .qtf_job import QTFJob

from typing import Any, Dict
from math import log2
from collections import Counter

logger = logging.getLogger(__name__)


class QTFQasmSimulator(Backend):
    """A Tensorflow based Qasm Simulator for Qiskit

    Parameters
    ----------
    Backend : qiskit.providers.BackendV1
        The QTFQasmSimulator is derived from BackendV1
    """

    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()["memory"] * (1024 ** 3) / 16))
    configuration = {
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

    DEFAULT_OPTIONS = {"initial_statevector": None, "chop_threshold": 1e-15}

    SHOW_FINAL_STATE = False  # noqa

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self.configuration)
            ),
            provider=provider,
            **fields
        )

        self._local_random = np.random.RandomState()  # TODO Tensorflow equivalent
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.options.get("initial_statevector")
        self._chop_threshold = self.options.get("chop_threashold")
        self._qobj_config = None
        # TEMP
        self._sample_measure = False

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024, memory=False, initial_statevector=None, chop_threshold=1e-15
        )

    def run(self, qobj: qobj.Qobj, **backend_options) -> QTFJob:
        """Parse and run a Qobj

        Parameters
        ----------
        qobj : Qobj
            The Qobj payload for the experiment
        backend_options : dict
            backend options

        Returns
        -------
        QTFJob
            An instance of the QTFJob (derived from JobV1) with the result

        Raises
        ------
        QiskitError
            Support for Pulse Jobs is not implemented

        Notes
        -----
        backend_options: Is a dict of options for the backend. It may contain
                * "initial_statevector": vector_like

        The "initial_statevector" option specifies a custom initial
        initial statevector for the simulator to be used instead of the all
        zero state. This size of this vector must be correct for the number
        of qubits in all experiments in the qobj.

        Example::

            backend_options = {
                "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),
            }
        """

        if isinstance(qobj, (QuantumCircuit, list)):
            qobj = assemble(qobj, self, **backend_options)
            qobj_options = qobj.config
        elif isinstance(qobj, qobj.PulseQobj):
            raise QiskitError("Pulse jobs are not accepted")
        else:
            qobj_options = qobj.config
        self._set_options(qobj_config=qobj_options, backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = QTFJob(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj

        Parameters
        ----------
        job_id : str
            unique id for the job
        qobj : Qobj
            job description

        Returns
        -------
        Result
            Result object
        """
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, "memory", False)
        self._qobj_config = qobj.config
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {
            "backend_name": self.name(),
            "backend_version": self._configuration.backend_version,
            "qobj_id": qobj.qobj_id,
            "job_id": job_id,
            "results": result_list,
            "status": "COMPLETED",
            "success": True,
            "time_taken": (end - start),
            "header": qobj.header.to_dict(),
        }

        return Result.from_dict(result)

    def run_experiment(self, experiment: QasmQobjExperiment) -> Dict[str, Any]:
        """Run an experiment (circuit) and return a single experiment result

        Parameters
        ----------
        experiment : QasmQobjExperiment
            experiment from qobj experiments list

        Returns
        -------
        Dict[str, Any]
            A result dictionary which looks something like::
            {
            "name": name of this experiment (obtained from qobj.experiment header)
            "seed": random seed used for simulation
            "shots": number of shots used in the simulation
            "data":
                {
                "counts": {'0x9: 5, ...},
                "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                },
            "status": status string for the simulation
            "success": boolean
            "time_taken": simulation time of this single experiment
            }

        Raises
        ------
        QTFError
            If an error occured
        """
        start = time.time()
        self._number_of_qubits = experiment.config.n_qubits
        self._number_of_cmembits = experiment.config.memory_slots
        self._statevector = 0  # TODO Convert to tf.Variable
        self._classical_memory = 0
        self._classical_register = 0
        self._sample_measure = False
        global_phase = experiment.header.global_phase
        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()
        # Get the seed looking in circuit, qobj, and then random.
        if hasattr(experiment.config, "seed_simulator"):
            seed_simulator = experiment.config.seed_simulator
        elif hasattr(self._qobj_config, "seed_simulator"):
            seed_simulator = self._qobj_config.seed_simulator
        else:
            # For compatibility on Windows force dyte to be int32
            # and set the maximum value to be (2 ** 31) - 1
            seed_simulator = np.random.randint(
                2147483647, dtype="int32"
            )  # TODO Convert to tf.random

        self._local_random.seed(seed=seed_simulator)
        # Check if measure sampling is supported for current circuit
        self._validate_measure_sampling(experiment)

        # List of final counts for all shots
        memory = []
        # Check if we can sample measurements, if so we only perform 1 shot
        # and sample all outcomes from the final state vector
        if self._sample_measure:
            shots = 1
            # Store (qubit, cmembit) pairs for all measure ops in circuit to
            # be sampled
            measure_sample_ops = []
        else:
            shots = self._shots
        for _ in range(shots):
            self._initialize_statevector()
            # apply global_phase
            self._statevector *= np.exp(1j * global_phase)  # TODO Convert to tf.exp
            # Initialize classical memory to all 0
            self._classical_memory = 0
            self._classical_register = 0
            for operation in experiment.instructions:
                conditional = getattr(operation, "conditional", None)
                if isinstance(conditional, int):
                    conditional_bit_set = (self._classical_register >> conditional) & 1
                    if not conditional_bit_set:
                        continue
                elif conditional is not None:
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_memory & mask
                        while (mask & 0x1) == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue

                # Check if single  gate
                if operation.name == "unitary":
                    qubits = operation.qubits
                    gate = operation.params[0]
                    self._add_unitary(gate, qubits)
                elif operation.name in ("U", "u1", "u2", "u3"):
                    params = getattr(operation, "params", None)
                    qubit = operation.qubits[0]
                    gate = single_gate_matrix(operation.name, params)  # TODO Implement
                    self._add_unitary(gate, [qubit])
                # Check if CX gate
                elif operation.name in ("id", "u0"):
                    pass
                elif operation.name in ("CX", "cx"):
                    qubit0 = operation.qubits[0]
                    qubit1 = operation.qubits[1]
                    gate = cx_gate_matrix()  # TODO Implement
                    self._add_unitary(gate, [qubit0, qubit1])
                # Check if reset
                elif operation.name == "reset":
                    qubit = operation.qubits[0]
                    self._add_qasm_reset(qubit)
                # Check if barrier
                elif operation.name == "barrier":
                    pass
                # Check if measure
                elif operation.name == "measure":
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = (
                        operation.register[0]
                        if hasattr(operation, "register")
                        else None
                    )

                    if self._sample_measure:
                        # If sampling measurements record the qubit and cmembit
                        # for this measurement for later sampling
                        measure_sample_ops.append((qubit, cmembit))
                    else:
                        # If not sampling perform measurement as normal
                        self._add_qasm_measure(qubit, cmembit, cregbit)
                elif operation.name == "bfunc":
                    mask = int(operation.mask, 16)
                    relation = operation.relation
                    val = int(operation.val, 16)

                    cregbit = operation.register
                    cmembit = operation.memory if hasattr(operation, "memory") else None

                    compared = (self._classical_register & mask) - val

                    if relation == "==":
                        outcome = compared == 0
                    elif relation == "!=":
                        outcome = compared != 0
                    elif relation == "<":
                        outcome = compared < 0
                    elif relation == "<=":
                        outcome = compared <= 0
                    elif relation == ">":
                        outcome = compared > 0
                    elif relation == ">=":
                        outcome = compared >= 0
                    else:
                        raise QTFError("Invalid boolean function relation.")

                    # Store outcome in register and optionally memory slot
                    regbit = 1 << cregbit
                    self._classical_register = (
                        self._classical_register & (~regbit)
                    ) | (int(outcome) << cregbit)
                    if cmembit is not None:
                        membit = 1 << cmembit
                        self._classical_memory = (
                            self._classical_memory & (~membit)
                        ) | (int(outcome) << cmembit)
                else:
                    backend = self.name()
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise QTFError(err_msg.format(backend, operation.name))

            # Add final creg data to memory list
            if self._number_of_cmembits > 0:
                if self._sample_measure:
                    # If sampling we generate all shot samples from the final statevector
                    memory = self._add_sample_measure(measure_sample_ops, self._shots)
                else:
                    # Turn classical_memory (int) into bit string and pad zero for unused cmembits
                    outcome = bin(self._classical_memory)[2:]
                    memory.append(hex(int(outcome, 2)))

        # Add data
        data = {"counts": dict(Counter(memory))}
        # Optionally add memory list
        if self._memory:
            data["memory"] = memory
        # Optionally add final statevector
        if self.SHOW_FINAL_STATE:
            data["statevector"] = self._get_statevector()
            # Remove empty counts and memory for statevector simulator
            if not data["counts"]:
                data.pop("counts")
            if "memory" in data and not data["memory"]:
                data.pop("memory")
        end = time.time()
        return {
            "name": experiment.header.name,
            "seed_simulator": seed_simulator,
            "shots": self._shots,
            "data": data,
            "status": "DONE",
            "success": True,
            "time_taken": (end - start),
            "header": experiment.header.to_dict(),
        }

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas."""
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise QTFError(
                "Number of qubits {} ".format(n_qubits)
                + "is greater than maximum ({}) ".format(max_qubits)
                + 'for "{}".'.format(self.name())
            )
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning(
                    'No classical registers in circuit "%s", ' "counts will be empty.",
                    name,
                )
            elif "measure" not in [op.name for op in experiment.instructions]:
                logger.warning(
                    'No measurements in circuit "%s", '
                    "classical register will remain all zeros.",
                    name,
                )
