# Публикации и источники

Проверено 2026-10-28. Здесь используется **JCR / Web of Science**, не SJR.
Для сопоставимости приведён **год метрик 2024**, а не квартиль в год выхода
каждой статьи. Категория также существенна: у одного журнала может быть
несколько квартилей. Прямой доступ к подписному интерфейсу Clarivate
не использовался; источники рейтинга — открытые университетские каталоги
и сообщение издателя с данными JCR.

## Квартили журналов

| Журнал | JCR 2024 | Категория | Источник |
|---|---|---|---|
| IEEE Transactions on Pattern Analysis and Machine Intelligence | Q1 | Computer Science, Artificial Intelligence | [Библиотека Wuhan University](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=6099) |
| Computer Vision and Image Understanding | Q2 | Computer Science, Artificial Intelligence | [Библиотека Wuhan University](https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=3400) |
| ACM Transactions on Graphics | Q1 | Computer Science, Software Engineering | [Библиотека Wuhan University](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=4612) |
| PLOS ONE | Q2 | Multidisciplinary Sciences | [Каталог Comillas IIT, таблица JCR-JIF](https://www.iit.comillas.edu/publicacion/info_revista/en/215/PLoS_One) |
| Sensors | Q2 | Instruments and Instrumentation | [Издатель, сообщение о JCR 2024 от 2025-06-19](https://www.mdpi.com/about/announcements/12201) |

Эти данные пригодны для предварительного отбора. Для формального требования
организации нужно получить запись JCR за требуемый ею год и категорию.
Квартиль статьи 2026 года здесь не называется «JCR 2026».

## OpenPose

Cao Z., Hidalgo G., Simon T., Wei S.-E., Sheikh Y.
**OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields.**
IEEE TPAMI, 43(1), 172–186, 2021; online publication 2019.
[DOI: 10.1109/TPAMI.2019.2929257](https://doi.org/10.1109/TPAMI.2019.2929257).

Это журнальная статья о детекторе. Она не является валидацией всей нашей
системы из сетевых камер, триангуляции, восстановления вращений и BVH-потока.
В [документации 3D-модуля](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/3d_reconstruction_module.md)
описаны штатные синхронизированные FLIR-камеры, пользовательский источник кадров
и прекращение активной поддержки этого модуля.

## Hi-ROS

Guidolin M., Tagliapietra L., Menegatti E., Reggiani M.
**Hi-ROS: Open-source multi-camera sensor fusion for real-time people tracking.**
Computer Vision and Image Understanding, 232, 103694, 2023.
[DOI: 10.1016/j.cviu.2023.103694](https://doi.org/10.1016/j.cviu.2023.103694).

Система объединяет и фильтрует готовые 3D-позы. В экспериментальной проверке
использовались пять синхронизированных Azure Kinect и их body-tracking SDK.
Это не готовый пакет инференса и триангуляции для произвольных RGB-камер.
[Код](https://github.com/HiROS-unipd),
[описание Skeleton Merger](https://github.com/HiROS-unipd/skeleton_merger),
[полный текст в репозитории университета](https://www.research.unipd.it/retrieve/5ea18a8f-6ab0-491c-8e49-8da3cbc6c1f8/1-s2.0-S1077314223000747-main.pdf).

## XNect

Mehta D. et al.
**XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera.**
ACM Transactions on Graphics, 39(4), 2020.
[DOI: 10.1145/3386569.3392410](https://doi.org/10.1145/3386569.3392410).

Несмотря на представление на SIGGRAPH, работа опубликована в журнале TOG.
Метод монокамерный, выдаёт скелетные углы. На
[странице авторов](https://vcai.mpi-inf.mpg.de/projects/XNect/)
есть модель и C++-библиотека; загрузка требует регистрации.
Доступность библиотеки не подтверждает наличие готового BVH-передатчика
или совместимость с текущим серверным окружением.

## MocapNET

Оригинальные публикации, перечисленные в
[репозитории](https://github.com/FORTH-ModelBasedTracker/MocapNET)
и [ветке mnet4](https://github.com/FORTH-ModelBasedTracker/MocapNET/tree/mnet4),
относятся к BMVC, ICPR и ICCV Workshops. Их нельзя называть журнальными Q1/Q2.

Отдельно опубликована независимая работа:

Ienaga N., Sekine K.
**Comparison of deep learning-based three-dimensional human pose estimation
methods with motion capture for gesture research.**
PLOS ONE, 21(4), e0347288, 2026.
[DOI: 10.1371/journal.pone.0347288](https://doi.org/10.1371/journal.pone.0347288).

Она рассматривает в том числе связку RTMPose + MocapNET v2.
Это исследование применения конкретной связки, не авторская журнальная статья
о MocapNET v4 и не проверка live-сервера с несколькими камерами.

## Pose2Sim

Pagnon D., Domalain M., Reveret L.
**Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1:
Robustness.** Sensors, 21(19), 6530, 2021.
[DOI: 10.3390/s21196530](https://doi.org/10.3390/s21196530).

**Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2:
Accuracy.** Sensors, 22(7), 2712, 2022.
[DOI: 10.3390/s22072712](https://doi.org/10.3390/s22072712).

Исходная цепочка использовала OpenPose и OpenSim. Исследования ограничены
изученными движениями и конфигурациями; численные ошибки нельзя переносить
на наши 2–3 камеры и жесты верхней части тела. В частности, в Part 2
описаны ограничения точности верхней части тела.

В [актуальном коде](https://github.com/perfanalytics/pose2sim) встроен RTMPose,
а полный покадровый live-тракт отмечен как будущая работа. Наличие окна
с детекциями или быстрого инференса отдельного кадра не означает готового
многокамерного live-сервера.

## Captury

**Concurrent Validation of a Multi-Camera Markerless Motion Capture System
Against Inertial Sensors for Upper- and Lower-Limb Joint Kinematics.**
Sensors, 26(17), 5492, 2026.
[DOI: 10.3390/s26175492](https://doi.org/10.3390/s26175492).

Исследование проверяет **CapturyStudio** относительно Xsens.
Оно не доказывает точность и задержку **CapturyLive** на нашей установке.
[Описание CapturyLive](https://captury.com/real-time-processing/) отдельно
перечисляет live-протоколы, экспорт файлов и конфигурации от четырёх камер.

## FreeMoCap, RTMPose и сетевой ввод

- [Архитектура FreeMoCap v2](https://github.com/freemocap/freemocap/blob/main/freemocap-docs/docs/architecture/backend-mocap.mdx): готовый инференс, триангуляция и live-обработка; документ относится к меняющейся ветке `main`.
- [FAQ v2](https://docs.freemocap.org/guides/faq/): live-скелет и обработка записей.
- [WebSocket API](https://docs.freemocap.org/reference/websocket-api/): внутренний изменяемый протокол, включая подтверждения кадров клиентом.
- [GPU](https://docs.freemocap.org/guides/gpu-setup/): CUDA-вариант для NVIDIA на Linux/Windows.
- [Камеры](https://docs.freemocap.org/guides/camera-setup/): USB и включение обнаружения виртуальных камер.
- [BVH/FBX](https://docs.freemocap.org/guides/export-formats/): экспорт через Blender, не обещание сетевого BVH-протокола.
- [Релизы](https://github.com/freemocap/freemocap/releases): на дату проверки опубликован v2.0.0-alpha.25; наличие release не гарантирует совпадения с документацией `main`.
- [rtmlib](https://github.com/Tau-J/rtmlib): готовые модели, Python-инференс, примеры камеры/видео и ускорения; не полный mocap-сервер.
- [Запрос RTSP-входа](https://github.com/freemocap/freemocap/issues/684): дополнительный контекст, не доказательство реализации функции.
- [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback): подача декодированного видео как виртуального устройства на Linux.

Публикацию о детекторе или стороннюю оценку RTMPose нельзя представлять
как проверку всего FreeMoCap v2. Обучение стандартных моделей с нуля
не требуется; совместимость модели и backend проверяется при развёртывании.

## MocapForAll

[Руководство разработчика](https://akiya-research-institute.github.io/MocapForAll-Manual/en/)
описывает многокамерный live-захват, сетевой VMC/VMT и BVH/FBX-файлы.
Подходящая журнальная публикация Q1/Q2 в этом исследовании не подтверждена.
