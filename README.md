# Структура файлов
```textgit 
ImuMotionCapture/
├── firmware/                     #  Заготовка под прошивки МК
│   ├── platforms/
│   │   ├── esp32/
│   │   └── arduino/
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

## Blender visualize
### 1-segment motion
В репозитории в /host/blender прикреплен блендер-проект для визуализации данных в реальном времени(`Human_spine.blend1`). Скрипт посылает команду для начала передачи данных. Контроллер передает данные в формате квантерниона вращения. Скетч `MPU6050_DMP_cmd.ino` принимает команды с хостового компьтера, осуществляет калибровку передачу углов с digital motion processor. При запуске скрипта получаем четкое движение без видимых задержек левого предплечья.

### 4-segment motion
В репозитории в /host/blender прикреплен блендер-проект для визуализации данных в реальном времени(`Human_spine_N_senseros.blend1`). Скрипт посылает команду для начала передачи данных. Контроллер передает данные в формате квантерниона вращения. Скетч `DMP_with_TCA9548A.ino` принимает команды с хостового компьтера, осуществляет калибровку передачу углов с digital motion processor. Получаем отклик со всех конечностей на движения датчиков с видимой задержкой в 0.5-1.5 секунды.

TODO:
1. Согласовать оси контроллера и объекта управления!(DONE)
2. Расширить на множетство датчиков(done): Расширено до 4х датчиков.
3. Настройка системы визуализации квантернионов.
4. Организация данных захвата движений.
5. Решить проблему наличия видимой задержки.(DONE)
6. Увеличить частоту дискретизации.(DONE)
7. Провести оценки погрешностей захвата готовой системы.
