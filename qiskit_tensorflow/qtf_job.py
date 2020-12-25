from qiskit.providers import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result


class QTFJob(Job):
    _async = False

    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self) -> None:
        """Submit a job to the simulator"""
        return

    def result(self) -> Result:
        """Return the result of the job

        Returns
        -------
        Result
            Result of the job simulation
        """
        return self._result

    def status(self) -> JobStatus:
        """Return job status

        Returns
        -------
        JobStatus
            Status of the job
        """
        return JobStatus.DONE
