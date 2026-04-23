import re
import numpy as np

def extract_floats(line: str) -> np.ndarray | None:
    """Оставляет только числа, пробелы и знаки минуса. Возвращает массив или None."""
    cleaned = re.sub(r'[^0-9.\- ]', ' ', line)
    parts = cleaned.split()
    if len(parts) < 9:
        return None
    try:
        return np.array([float(x) for x in parts[:9]])
    except ValueError:
        return None