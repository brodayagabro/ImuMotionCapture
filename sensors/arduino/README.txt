├── complementary_filter_example/ - пример чтения с комплиментароного фильтра
│   ├── 06112025_angles_plus_mag.png
│   ├── 06112025_horizintal_roll_pitch_mess.jpg
│   ├── 06112025_horizontal_roll_pitch_mess.png
│   ├── 06112025_moving.png
│   ├── 06112025_vert.jpg
│   ├── complementary_filter_example.ino
│   ├── MPU9250connect.jpg
│   └── Pasted image 20251028123107.png
├── DMP_with_TCA9548A/ - сбор данных с нескольких датчиком(dmp) с помощью мультиплексора TCA9548A
│   └── DMP_with_TCA9548A.ino
├── MPU6050_DMP6/ - Пример работы с DMP из используемой библиотеки+команды START/STOP/CLIB
│   ├── MPU6050_DMP6.ino
│   └── Processing/
│       └── MPUTeapot/
│           └── MPUTeapot.pde
├── MPU6050_DMP6_cmd/ - DMP+START/STOP/CALIB+TCA9548a on 1st channel
│   ├── MPU6050_DMP6_cmd.ino
│   └── Processing/
│       └── MPUTeapot/
│           └── MPUTeapot.pde
├── README.txt
└── scan_bus/ - скетч для сканирования I2c-шины с мультиплексором показывает все адреса на каналах мультиплексора
    └── scan_bus.ino


TODO
1) Add command to change frequency
2) Mix MPU6050_DMP6_cmd with PCA9548A script(done)
3) Add posibility of bluetooth/wifi connection
4) Add data-base implementation
