import sys
import numpy as np
from dataLoad import datLoDe
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QComboBox, QTableWidget, QWidget

class segment_tsg(QWidget):
    def __init__(self):
        super().__init__()
        self.file_list = []
    
    def GUI(self):
        self.setWindowTitle("Time Series Segmentation App")
        self.setGeometry(100, 100, 825, 300)
        
        #display block
        self.len_sig_label = QLabel("Sample(s):",self)
        self.len_sig_label.setGeometry(5,210,100,20)

        self.len_sig_value = QLabel("0",self)
        self.len_sig_value.setGeometry(105,210,100,20)
        
        self.max_value_label = QLabel("Max:",self)
        self.max_value_label.setGeometry(5,240,100,20)
        
        self.max_value = QLabel("None",self)
        self.max_value.setGeometry(105,240,100,20)
        
        self.min_value_label = QLabel("Min:",self)
        self.min_value_label.setGeometry(5,270,100,20)
        
        self.min_value = QLabel("None",self)
        self.min_value.setGeometry(105,270,100,20)

        #segment block
        self.next = QPushButton("Next",self)
        self.next.setGeometry(220,210,50,20)

        self.back = QPushButton("Back",self)
        self.back.setGeometry(280,210,50,20)

        self.home = QPushButton("Home",self)
        self.home.setGeometry(340,210,50,20)

        self.start_seg_lable = QLabel("Start:",self)
        self.start_seg_lable.setGeometry(220,240,50,20)

        self.end_seg_label = QLabel("End:",self)
        self.end_seg_label.setGeometry(280,240,50,20)

        self.step_label = QLabel("Step:",self)
        self.step_label.setGeometry(340,240,50,20)

        self.start_seg = QLineEdit(self)
        self.start_seg.setGeometry(220,270,50,20)

        self.end_seg = QLineEdit(self)
        self.end_seg.setGeometry(280,270,50,20)

        self.step = QLineEdit(self)
        self.step.setGeometry(340,270,50,20)

        #control block
        self.link_line = QLineEdit(self)
        self.link_line.setGeometry(410,210,150,20)

        self.browse = QPushButton("Browse",self)
        self.browse.setGeometry(570,210,50,20)

        self.type_label = QLabel("Type:",self)
        self.type_label.setGeometry(410,240,50,20)

        self.type = QComboBox(self)
        self.type.setGeometry(520,240,100,20)

        self.save = QPushButton("Save",self)
        self.save.setGeometry(570,270,50,20)

        #data list
        self.data_list = QTableWidget(self)
        self.data_list.setGeometry(630,5,185,290)

        #display signal
        self.figure, self.ax = plt.subplots(figsize=(8, 6))  # Adjust size here
        self.canvas = FigureCanvas(self.figure)
        self.canvas.move(5,5)
        self.canvas.resize(620,200)
        # self.addToolBar(NavigationToolbar(self.canvas, self))

    def get_link(self):
        pass

    def load_data(self):
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = segment_tsg()
    window.GUI()
    window.show()
    sys.exit(app.exec_())
