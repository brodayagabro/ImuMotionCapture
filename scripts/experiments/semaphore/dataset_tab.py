"""Read-only dataset overview and background rebuild of derived CSV tables."""
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .config import ROOT
from .dataset import build_dataset


class _Export(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, root, parent):
        super().__init__(parent)
        self.root = root

    def run(self):
        try:
            roots = [ROOT / "recordings"] if self.root == (ROOT / "records" / "semaphore").resolve() else []
            self.result.emit(build_dataset(self.root, roots))
        except Exception as error:
            self.error.emit(str(error))


class DatasetTab(QWidget):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root).resolve()
        self.worker = None
        self._again = False
        layout = QVBoxLayout(self)
        description = QLabel("Одна строка — связь участника и запуска эксперимента с записью костюмов.\n"
            "CSV содержат также параметры, пути, попытки слов и интервалы букв.\n"
            "После STOP REC обновите реестр, чтобы увидеть итоговый статус записи.")
        description.setWordWrap(True)
        layout.addWidget(description)
        bar = QHBoxLayout()
        self.refresh = QPushButton("Обновить реестр")
        self.refresh.clicked.connect(self.request_refresh)
        bar.addWidget(self.refresh)
        open_folder = QPushButton("Папка таблиц")
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root / "dataset"))))
        bar.addWidget(open_folder)
        layout.addLayout(bar)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Участник", "Начало", "Режим", "Эксперимент", "Запись", "Связь", "Папка костюмов"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self.status = QLabel("Таблицы: " + str(self.root / "dataset"))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @property
    def busy(self):
        return self.worker is not None

    def request_refresh(self):
        if self.busy:
            self._again = True
            return
        self.refresh.setEnabled(False)
        self.status.setText("Сопоставление событий и записей…")
        self.worker = _Export(self.root, self)
        self.worker.result.connect(self._show_result)
        self.worker.error.connect(lambda error: self.status.setText("Не удалось обновить реестр: " + error))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _show_result(self, result):
        labels = {"completed": "Завершён", "stopped": "Остановлен", "interrupted": "Прерван",
                  "running": "Идёт", "recording": "Идёт запись", "complete": "Полная", "incomplete": "Неполная",
                  "error": "Ошибка", "unknown": "Неизвестно", "events_verified": "Подтверждена",
                  "ack_only": "Есть ACK", "unconfirmed": "Не подтверждена", "missing_recording": "Папка не найдена",
                  "ambiguous_recording": "Неоднозначная", "offline": "Без рекордера", "unmatched": "Не найдена"}
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(len(result["index"]))
        for i, row in enumerate(result["index"]):
            values = [row["participant_id"], row["started_at"], "Тренировка" if row["training_trial"] else "Эксперимент",
                      labels.get(row["presenter_status"], row["presenter_status"]),
                      labels.get(row["recording_status"], row["recording_status"]),
                      labels.get(row["link_status"], row["link_status"]), row["recording_dir"]]
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self.table.setUpdatesEnabled(True)
        counts = result["counts"]
        self.status.setText(f"Сессий: {result['sessions']} · связей: {counts['dataset_index.csv']} · "
                            f"удержаний: {counts['letter_holds.csv']} · замечаний: {len(result['warnings'])}\n"
                            + result["output_dir"] + ("\nПодробности замечаний — report.json" if result["warnings"] else ""))

    def _finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.refresh.setEnabled(True)
        if self._again:
            self._again = False
            self.request_refresh()
