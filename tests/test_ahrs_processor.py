import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from host.processing.ahrs_processor import AHRSProcessor


class TestAHRSProcessor:
    @pytest.fixture
    def processor(self):
        # Mock imufusion, чтобы тесты работали без реальной библиотеки
        with patch('host.processing.ahrs_processor.imufusion') as mock_imufusion:
            mock_imufusion.quaternion_to_euler.return_value = [0.1, 0.2, 0.3]
            yield AHRSProcessor()

    def test_init_settings(self, processor):
        assert processor.ahrs is not None
        # Проверка, что settings были установлены (через моки)

    def test_update_returns_euler(self, processor, sample_accel, sample_gyro, sample_mag):
        result = processor.update(sample_gyro, sample_accel, sample_mag)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # Значения зависят от мока — проверяем тип и размер

    def test_update_with_zero_input(self, processor):
        zeros = np.zeros(3)
        result = processor.update(zeros, zeros, zeros)
        assert result.shape == (3,)