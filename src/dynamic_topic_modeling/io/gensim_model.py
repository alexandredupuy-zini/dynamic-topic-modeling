from typing import Any, Union, Dict
from os.path import isfile
import sys
from kedro.io import AbstractDataSet
from gensim.models import LdaModel

class GensimModel(AbstractDataSet):

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
        sys.path.insert(0, './src/dynamic_topic_modeling/pipelines/machine_learning')
        return LdaModel.load(self._filepath)

    def _save(self, model) -> None:
        model.save(self._filepath)

    def _exists(self) -> bool:
        return isfile(self._filepath)
