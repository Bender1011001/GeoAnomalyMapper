import pytest
from gam.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    IngestionError,
    PreprocessingError,
    ModelingError,
    VisualizationError,
    PipelineError,
    GAMError,
)

@pytest.mark.parametrize(
    "exception_class",
    [ConfigurationError, DataValidationError, IngestionError, PreprocessingError, ModelingError, VisualizationError, PipelineError, GAMError],
)
def test_custom_exceptions_can_be_raised(exception_class):
    """
    Verify that custom exceptions can be raised and caught.
    """
    with pytest.raises(exception_class, match="This is a test error"):
        raise exception_class("This is a test error")

def test_error_hierarchy():
    """
    Verify that all custom exceptions inherit from GAMError.
    """
    assert issubclass(ConfigurationError, GAMError)
    assert issubclass(DataValidationError, GAMError)
    assert issubclass(IngestionError, GAMError)
    assert issubclass(PreprocessingError, GAMError)
    assert issubclass(ModelingError, GAMError)
    assert issubclass(VisualizationError, GAMError)
    assert issubclass(PipelineError, GAMError)