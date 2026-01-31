# UI Stylesheets for Face System

def get_stylesheet(theme="dark"):
    """
    Returns the QSS stylesheet based on the selected theme.
    Supported themes: 'dark', 'light'
    """
    
    # Common font settings
    common = """
        * { font-family: 'Noto Sans CJK TC', 'Microsoft JhengHei', sans-serif; }
    """
    
    if theme == "dark":
        return common + """
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #3b3b3b; border: 1px solid #555; padding: 5px; color: white; }
            QTableWidget { background-color: #3b3b3b; gridline-color: #555; color: white; }
            QHeaderView::section { background-color: #444; padding: 4px; border: 1px solid #555; color: white; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #3b3b3b; color: #aaa; padding: 8px 20px; }
            QTabBar::tab:selected { background: #505050; color: white; }
            QGroupBox { border: 1px solid #555; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }
            QPushButton { background-color: #555; color: white; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #666; }
            QMessageBox { background-color: #2b2b2b; color: white; }
            QMessageBox QLabel { color: white; }
            QInputDialog { background-color: #2b2b2b; color: white; }
            QInputDialog QLabel { color: white; }
            QComboBox { background-color: #3b3b3b; border: 1px solid #555; padding: 5px; color: white; }
            QComboBox::drop-down { border: 0px; }
        """
    else:
        # Light Theme (Modern Light)
        return common + """
            QWidget { background-color: #f0f0f0; color: #000000; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #ffffff; border: 1px solid #ccc; padding: 5px; color: black; }
            QTableWidget { background-color: #ffffff; gridline-color: #ccc; color: black; }
            QHeaderView::section { background-color: #e0e0e0; padding: 4px; border: 1px solid #ccc; color: black; }
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab { background: #e0e0e0; color: #333; padding: 8px 20px; }
            QTabBar::tab:selected { background: #ffffff; color: black; border-bottom: 2px solid #4CAF50; }
            QGroupBox { border: 1px solid #ccc; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; background-color: #f0f0f0; }
            QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #d0d0d0; }
            QMessageBox { background-color: #f0f0f0; color: black; }
            QMessageBox QLabel { color: black; }
            QInputDialog { background-color: #f0f0f0; color: black; }
            QInputDialog QLabel { color: black; }
            QComboBox { background-color: #ffffff; border: 1px solid #ccc; padding: 5px; color: black; }
        """
