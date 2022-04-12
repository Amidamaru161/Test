

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np


class Ui_Predictor(object):
    def setupUi(self, Predictor):
        Predictor.setObjectName("Predictor")
        Predictor.resize(811, 612)
        self.centralwidget = QtWidgets.QWidget(Predictor)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 811, 611))
        self.widget.setStyleSheet("QWidget#widget{\n"
"background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0.893, y2:0.863636, stop:0 rgba(116, 224, 255, 255), stop:0.875706 rgba(255, 227, 227, 255));}")
        self.widget.setObjectName("widget")
        self.label_welcome = QtWidgets.QLabel(self.widget)
        self.label_welcome.setGeometry(QtCore.QRect(330, 20, 151, 51))
        self.label_welcome.setStyleSheet("font: 28pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_welcome.setTextFormat(QtCore.Qt.AutoText)
        self.label_welcome.setObjectName("label_welcome")
        self.label_jpg = QtWidgets.QLabel(self.widget)
        self.label_jpg.setGeometry(QtCore.QRect(80, 120, 641, 421))
        self.label_jpg.setText("")
        self.label_jpg.setPixmap(QtGui.QPixmap("C:/Users/bodya/OneDrive/Изображения/pix.png"))
        self.label_jpg.setObjectName("label_jpg")
        self.btn_predict = QtWidgets.QPushButton(self.widget)
        self.btn_predict.setGeometry(QtCore.QRect(400, 550, 321, 41))
        self.btn_predict.setStyleSheet("border-radius:20px; background-color: rgb(248, 220, 150);\n"
"font: 28pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.btn_predict.setObjectName("btn_predict")
        self.label_text = QtWidgets.QLabel(self.widget)
        self.label_text.setGeometry(QtCore.QRect(150, 80, 511, 31))
        self.label_text.setStyleSheet("font: 18pt \"MS PGothic\" ;color:rgb(255, 255, 255)\n"
"\n"
"\n"
"\n"
"")
        self.label_text.setObjectName("label_text")
        self.line = QtWidgets.QLineEdit(self.widget)
        self.line.setGeometry(QtCore.QRect(90, 570, 251, 16))
        self.line.setText("")
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(80, 540, 271, 31))
        self.label.setStyleSheet("font: 18pt \"MS PGothic\" ;color:rgb(255, 100, 255)")
        self.label.setObjectName("label")
        Predictor.setCentralWidget(self.centralwidget)

        self.retranslateUi(Predictor)
        QtCore.QMetaObject.connectSlotsByName(Predictor)
        self.add_functions()
        
    def retranslateUi(self, Predictor):
        _translate = QtCore.QCoreApplication.translate
        Predictor.setWindowTitle(_translate("Predictor", "MainWindow"))
        self.label_welcome.setText(_translate("Predictor", "Welcome"))
        self.btn_predict.setText(_translate("Predictor", "Predict"))
        self.label_text.setText(_translate("Predictor", "This program predicts what is shown in the picture"))
        self.label.setText(_translate("Predictor", " Enter the picture number"))

    def add_functions(self):
        self.btn_predict.clicked.connect(self.write_number)    
        self.btn_predict.clicked.connect(self.result3)
        
    def result3(self):
        error=QMessageBox()
        error.setWindowTitle("Прогноз")
        #error.setText(f"С вероятностью {int(data[0][2]*100)}% на картинке изображен {data[0][1]}")
        error.setText(f"С вероятностью {int(self.data[0][2]*100)}% на картинке изображен {self.data[0][1]}")
        error.setIcon(QMessageBox.Information) 
        error.setStandardButtons(QMessageBox.Ok)         
        error.exec_()      

    def model(self,img_path):        
        model = VGG16(weights='imagenet')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x) 
        preds = model.predict(x)
        self.data=(decode_predictions (preds, top=1)[0]) 
        np.argmax(preds[0])
        
    
    def write_number(self):
        img_num=self.line.text()
        img_path = f"C:\\image\{str(img_num)}.jpg"    
        self.label_jpg.setPixmap(QtGui.QPixmap(img_path))
        self.model(img_path)
        
        
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Predictor = QtWidgets.QMainWindow()
    ui = Ui_Predictor()
    ui.setupUi(Predictor)
    Predictor.show()
    sys.exit(app.exec_())
