import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLineEdit

class TradingSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading System")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.layout.addWidget(self.log_display)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter command...")
        self.layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_input)
        self.layout.addWidget(self.send_button)

    def handle_input(self):
        command = self.input_field.text()
        self.log_display.append(f"User command: {command}")
        # Process the command here
        self.input_field.clear()

    def log(self, message):
        self.log_display.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = TradingSystemGUI()
    main_window.show()
    sys.exit(app.exec_())