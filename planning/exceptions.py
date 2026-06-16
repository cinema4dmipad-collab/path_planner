class ContourError(Exception):
    """Базовое исключение для ошибок контура."""

class EmptyFileError(ContourError):
    """Файл не содержит точек."""

class InsufficientPointsError(ContourError):
    """Недостаточно точек для построения контура."""

class InvalidFormatError(ContourError):
    """Некорректный формат данных в файле."""


class BridgePlanningError(ContourError):
    """Не удалось построить безопасный переход между сегментами траектории."""


class DXFError(Exception):
    """Базовое исключение для ошибок DXF."""
    pass

class DXFLoadError(DXFError):
    """Ошибка загрузки DXF-файла."""
    pass

class UnsupportedEntityError(DXFError):
    """Неподдерживаемый тип примитива DXF."""
    pass

class FileNotFoundError(DXFError):
    """Файл не найден."""
    pass