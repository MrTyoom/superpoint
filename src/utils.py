import shutil
from pathlib import Path


def make_dir(path: str | Path) -> Path:
    """Создает новую директорию,
    удаляя старое содержимое директории.
    """
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path
