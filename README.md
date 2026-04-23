# ImuMotionCapture
mocap-system/
├── schemas/                  # Общие схемы сообщений (FlatBuffers / Protobuf)
│   ├── sensor_data.fbs       # Сырые/откалиброванные данные датчика
│   ├── control_cmd.fbs       # Команды хост→костюм (старт, калибровка, частота)
│   └── generate.sh           # Скрипт генерации кода для FW и host
├── firmware/                 # Прошивки для бортовых МК
│   ├── platforms/            # Разделение по чипам
│   │   ├── esp32/
│   │   ├── nrf52/
│   │   └── stm32/
│   ├── lib/                  # Переиспользуемые библиотеки
│   │   ├── imu/              # Драйверы BNO085, ICM-42688 и т.д.
│   │   ├── math/             # Кватернионы, матрицы, комплементарный фильтр (легковесный)
│   │   ├── comms/            # Абстракция транспорта (UDP, BLE GATT, BLE Serial)
│   │   └── calib/            # Сбор смещений, температурная компенсация
│   └── tools/                # PlatformIO.ini, CMake, flash-скрипты, FW-CI
├── host/                     # Система на ПК
│   ├── ingestion/            # Приём данных
│   │   ├── adapters/         # udp_adapter.rs, ble_adapter.py/rs
│   │   ├── unifier.rs        # Сборка пакетов, таймстампы, seq-контроль
│   │   └── control_channel/  # Отправка команд на костюм (BLE GATT write / UDP)
│   ├── processing/           # EKF/Madgwick, калибровка, скелетная модель
│   ├── storage/              # TimescaleDB клиент, батчевая запись, экспорт
│   ├── viz/                  # Визуализация (PyQt/Three.js)
│   ├── api/                  # FastAPI, управление экспериментами, стрим
│   └── shared/               # Десериализация схем, логгер, конфиг, метрики
├── tests/                    # Mock-датчики, BLE-эмуляторы, интеграционные тесты
├── deploy/                   # docker-compose, systemd, RT-скрипты
└── docs/                     # Протоколы, схема калибровки, инструкции