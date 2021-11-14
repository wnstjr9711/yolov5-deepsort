import sys

from PySide2.QtWidgets import QApplication
from gui.mainwindow import MainWindow
import qasync

app = QApplication(sys.argv)
app.setStyle('Fusion')
display_size = app.primaryScreen().size()

window = MainWindow()
window.show()
window.setWindowTitle('verticV-Demo')

loop = qasync.QEventLoop(app)
loop.run_forever()
