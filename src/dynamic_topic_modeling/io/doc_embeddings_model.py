from typing import Any, Union, Dict
import sys
from os.path import isfile
from kedro.io import AbstractDataSet
from gensim.models import Doc2Vec

class DocEmbeddingsModel(AbstractDataSet):

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(
        self,
        filepath: str,
    ) -> None:
        """Creates a new instance of ``Torch_Model`` pointing to a concrete
        filepath.

        Args:
            filepath: path to a MmCorpus file.
        """
        self._filepath = filepath

    def _load(self):
        return Doc2Vec.load(self._filepath)

    def _save(self, model) -> None:
        model.save(self._filepath)

    def _exists(self) -> bool:
        return isfile(self._filepath)
