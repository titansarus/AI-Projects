import logging
import sys

from PyQt5.uic import loadUiType

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)

Ui_MainWindow, QMainWindow = loadUiType('dialog.ui')

from FirstSection.Regressor import Regressor_Simple


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
        self.plot_btn.clicked.connect(self.make_plot)
        self.predict_btn.clicked.connect(self.make_predict)
        self.regressor = Regressor_Simple()

        logTextBox = QTextEditLogger(self.log_lv)
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)

    def make_plot(self):
        fig = self.regressor.plot()
        self.addmpl(fig)

    def make_predict(self):
        x = float(self.pred_x_in.text())
        y = self.regressor.model.predict([x])[0]
        self.predict_y.setText(str(y))

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
        reg = self.regressor
        reg.NUMBER_OF_FOLDS = int(self.fold_lbl.text())
        reg.EPOCHS = int(self.epoch_lbl.text())
        reg.BATCH_SIZE = int(self.batch_lbl.text())
        reg.train_low = float(self.train_low_in.text())
        reg.train_high = float(self.train_high_in.text())
        reg.plot_low = float(self.plot_low_in.text())
        reg.plot_high = float(self.plot_high_in.text())
        reg.NUMBER_OF_SAMPLES = int(self.samples_in.text())
        reg.NUMBER_OF_SAMPLE_FOR_PLOT = int(self.plot_samples_in.text())
        reg.function = lambda x: eval(self.function_text.text())

    def make_dataset(self):
        self.regressor.initial_dataset_maker()

    def reset_functionality(self):
        self.plot_low_in.setText("0")
        self.plot_high_in.setText("4")
        self.train_low_in.setText("0.1")
        self.train_high_in.setText("2")
        self.samples_in.setText("20000")
        self.plot_samples_in.setText("200000")
        self.function_text.setText("np.sin(x)")
        self.epoch_sld.setValue(6)
        self.fold_sld.setValue(5)
        self.batch_sld.setValue(5)

    def learn_action(self):
        self.regressor.train()
        self.best_loss_lbl.setText(str(self.regressor.best_loss))
        self.avg_loss_lbl.setText(str(self.regressor.avg_loss))

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
