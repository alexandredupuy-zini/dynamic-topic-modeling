from typing import Any, Union, Dict
import tarfile
import re
from os.path import isfile

from kedro.io import AbstractDataSet

class TgzNipsDataSet(AbstractDataSet):

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(
        self,
        filepath: str,
    ) -> None:
        """Creates a new instance of ``TgzLocalDataSet`` pointing to a concrete
        filepath.

        Args:
            filepath: path to an Tgz file.
        """
        self._filepath = filepath

    def _load(self) -> list:
        url = self._filepath
        docs = []
        with tarfile.open(url, mode='r:gz') as tar:
            files = [
                m for m in tar.getmembers()
                if m.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', m.name)
            ]
            for member in sorted(files, key=lambda x: x.name):
                member_bytes = tar.extractfile(member).read()
                docs.append(member_bytes.decode('utf-8', errors='replace'))
        return docs

    def _save(self) -> None:
        pass

    def _exists(self) -> bool:
        return isfile(self._filepath)