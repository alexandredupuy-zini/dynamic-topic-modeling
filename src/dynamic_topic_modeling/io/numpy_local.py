from typing import Any, Union, Dict
from os.path import isfile
import numpy as np
from kedro.io import AbstractDataSet

class NumpyArray(AbstractDataSet):
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

    def _load(self) -> np.ndarray :
        return np.load(self._filepath)

    def _save(self, array : np.ndarray) -> None:
        np.save(self._filepath,array)

    def _exists(self) -> bool:
        return isfile(self._filepath)