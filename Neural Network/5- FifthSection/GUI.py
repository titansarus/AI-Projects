import logging
import sys

from PyQt5.uic import loadUiType

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from tensorflow import keras

Ui_MainWindow, QMainWindow = loadUiType('dialog.ui')

from FifthSection.Classifier import Image_Classifier


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QTextEdit()
        parent.addWidget(self.widget)
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum());

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.sldOnChangeLinear('epoch_lbl', 'epoch_sld')
        self.sldOnChangeLinear('fold_lbl', 'fold_sld')
        self.sldOnChangeExponential('batch_lbl', 'batch_sld', 2)
        self.reset_btn.clicked.connect(self.reset_functionality)
        self.set_btn.clicked.connect(self.set_values)
        self.learn_btn.clicked.connect(self.learn_action)
        self.make_btn.clicked.connect(self.make_dataset)
        self.predict_btn.clicked.connect(self.make_predict)
        self.save_btn.clicked.connect(self.save_model)
        self.load_btn.clicked.connect(self.load_model)
        self.random_btn.clicked.connect(self.random_number_for_predict)
        self.classifier = Image_Classifier()

        logTextBox = QTextEditLogger(self.log_lv)
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.INFO)

    def make_predict(self):
        x = int(self.pred_x_in.text())
        fig, answer = self.classifier.image_predict(x)
        self.predict_y.setText(str(answer))
        self.addmpl(fig)

    def addmpl(self, fig):
        for i in reversed(range(self.mplvl.count())):
            self.mplvl.itemAt(i).widget().setParent(None)
        self.canvas = FigureCanvas(fig)
        FigureCanvas.setSizePolicy(
            self.canvas, QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self.canvas)
        self.mplvl.addWidget(self.canvas)

        self.canvas.draw()

    def set_values(self):
        cls = self.classifier
        cls.NUMBER_OF_FOLDS = int(self.fold_lbl.text())
        cls.EPOCHS = int(self.epoch_lbl.text())
        cls.BATCH_SIZE = int(self.batch_lbl.text())

    def make_dataset(self):
        self.classifier.load_mnist()

    def save_model(self):
        self.classifier.model.save("trained_model")

    def random_number_for_predict(self):
        number = np.random.randint(0, 60000, 1)[0]
        self.pred_x_in.setText(str(number))

    def load_model(self):
        self.classifier.model = keras.models.load_model("trained_model")
        self.make_dataset()

    def reset_functionality(self):
        self.epoch_sld.setValue(6)
        self.fold_sld.setValue(5)
        self.batch_sld.setValue(5)

    def learn_action(self):
        self.classifier.train()
        self.best_loss_lbl.setText(str(self.classifier.best_loss))
        self.avg_loss_lbl.setText(str(self.classifier.avg_loss))

    def sldOnChangeLinear(self, lbl: str, sld: str):
        lbl1 = getattr(self, lbl)
        sld1 = getattr(self, sld)

        def func():
            lbl1.setText(str(sld1.value()))

        sld1.valueChanged.connect(func)

    def sldOnChangeExponential(self, lbl: str, sld: str, base=2):
        lbl1 = getattr(self, lbl)
        sld1 = getattr(self, sld)

        def func():
            lbl1.setText(str(base ** sld1.value()))

        sld1.valueChanged.connect(func)


if __name__ == '__main__':
    import sys
    from PyQt5 import QtGui, QtWidgets
    import numpy as np

    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
