from typing import Any, Union, Dict
from os.path import isfile
import scipy 
from kedro.io import AbstractDataSet

class ScipySparseMatrix(AbstractDataSet):
    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(
        self,
        filepath: str,
    ) -> None:
        """Creates a new instance of ``MmCorpusDataSet`` pointing to a concrete
        filepath.

        Args:
            filepath: path to a MmCorpus file.
        """
        self._filepath = filepath

    def _load(self) -> scipy.sparse.csr.csr_matrix:
        return scipy.sparse.load_npz(self._filepath)

    def _save(self, matrix : scipy.sparse.csr.csr_matrix) -> None:
        scipy.sparse.save_npz(self._filepath,matrix)

    def _exists(self) -> bool:
        return isfile(self._filepath)