from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class LargeTitleBar(QWidget):
    def __init__(
            self, title, parent=None, show_close=False, centered=False,
            font_size=20, align_bottom=False, height=76, theme="dark"):
        super().__init__(parent)
        self._drag_pos = None
        self.setFixedHeight(height)
        self.setObjectName("largeTitleBar")
        self._centered = centered

        layout = QHBoxLayout()
        layout.setContentsMargins(22, 0, 16, 0)
        layout.setSpacing(12)
        self.setLayout(layout)

        self.label = QLabel(title)
        self.label.setObjectName("largeTitleLabel")
        self.label.setFont(QFont("Microsoft JhengHei", font_size, QFont.Bold))
        title_color = "#111827" if theme == "light" else "#ffffff"
        self.label.setStyleSheet(
            f"color: {title_color}; font-size: {font_size}pt; font-weight: 700;")
        if centered and align_bottom:
            self.label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        elif centered:
            self.label.setAlignment(Qt.AlignCenter)
        else:
            self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(self.label, 1)

        if show_close:
            self.close_button = QPushButton("X")
            self.close_button.setObjectName("largeTitleCloseButton")
            self.close_button.setFixedSize(56, 56)
            if parent is None:
                close_action = self.close
            elif hasattr(parent, "reject"):
                close_action = parent.reject
            else:
                close_action = parent.close
            self.close_button.clicked.connect(close_action)
            layout.addWidget(self.close_button)

        if theme == "light":
            title_bar_style = """
            QWidget#largeTitleBar {
                background-color: #e5e7eb;
                border-bottom: 2px solid #cbd5e1;
            }
            QPushButton#largeTitleCloseButton {
                background-color: #f8fafc;
                color: #111827;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                font-size: 18pt;
                font-weight: 700;
                min-width: 56px;
                max-width: 56px;
                min-height: 56px;
                max-height: 56px;
                padding: 0;
            }
            QPushButton#largeTitleCloseButton:hover {
                background-color: #dc2626;
                color: #ffffff;
                border-color: #ffffff;
            }
            """
        else:
            title_bar_style = """
            QWidget#largeTitleBar {
                background-color: #111827;
                border-bottom: 2px solid #4b5563;
            }
            QPushButton#largeTitleCloseButton {
                background-color: #374151;
                color: #ffffff;
                border: 1px solid #6b7280;
                border-radius: 4px;
                font-size: 18pt;
                font-weight: 700;
                min-width: 56px;
                max-width: 56px;
                min-height: 56px;
                max-height: 56px;
                padding: 0;
            }
            QPushButton#largeTitleCloseButton:hover {
                background-color: #dc2626;
                border-color: #ffffff;
            }
            """
        self.setStyleSheet(title_bar_style)

    def set_title(self, title):
        self.label.setText(title)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            window = self.window()
            window.move(window.pos() + event.globalPos() - self._drag_pos)
            self._drag_pos = event.globalPos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)
