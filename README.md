# Структура файлов
```text
ImuMotionCapture/
├── firmware/                     #  Заготовка под прошивки МК
│   ├── platforms/
│   │   ├── esp32/
│   │   └── stm32/
│   ├── lib/
│   │   ├── imu/
│   │   ├── comms/
│   │   └── math/
│   └── src/
│       └── main.cpp              # ← Заглушка
├── host/                         #  Python-приложение (ПК)
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── config.py             #  Константы (COM, UDP, BAUD, SAMPLE_RATE)
│   │   └── data_parser.py        #  Очистка строк, извлечение 9 float
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── base.py           #  Абстрактный интерфейс бэкенда
│   │   │   ├── serial.py         #  Чтение COM-порта
│   │   │   └── udp.py            #  Чтение UDP-сокета
│   │   └── reader.py             #  Координатор потоков + callback-диспетчер
│   ├── processing/
│   │   ├── __init__.py
│   │   └── ahrs_processor.py     #  Обёртка над imufusion (AHRS/Euler)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db_client.py          #  Заготовка под asyncpg/TimescaleDB
│   └── viz/
│       ├── __init__.py
│       ├── main.py               #  Точка входа (запуск приложения)
│       ├── app.py                #  Основной мульти-режимный UI
│       ├── visualizer.py         #  Matplotlib 3D-куб + оси
│       └── single_sensor_viz.py  #  Отдельное приложение: Raw + Processed
├── schemas/
│   └── sensor_data.fbs           #  Схема данных (FlatBuffers/Protobuf)
├── tests/
│   └── test_parser.py            #  Мок-тесты для парсера/бэкендов
├── docs/
├── deploy/
├── requirements.txt              #  pip install -r requirements.txt
└── README.md
```
# Запуск тестов
```cmd
# 1. Установите dev-зависимости
pip install -r requirements-dev.txt

# 2. Запустите все тесты с покрытием
pytest

# 3. Запустить один файл
pytest tests/test_data_parser.py -v

# 4. Запустить с отчётом о покрытии (откроется htmlcov/index.html)
pytest --cov=host --cov-report=html

# 5. Запустить тесты с таймаутом 5 секунд на тест
pytest --timeout=5
```