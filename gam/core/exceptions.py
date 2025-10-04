class GAMError(Exception):
    "Base exception for all application-specific errors."
    pass


class ConfigurationError(GAMError):
    pass


class DataValidationError(GAMError):
    pass


class IngestionError(GAMError):
    pass


class PreprocessingError(GAMError):
    pass


class ModelingError(GAMError):
    pass


class VisualizationError(GAMError):
    pass


class PipelineError(GAMError):
    pass