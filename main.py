import ctypes
import sys
import time
import os

import cv2
import numpy as np

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QThread
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from custom.graphicsView import GraphicsView
from custom.listWidgets import *
from custom.stackedWidget import *
from custom.treeView import FileSystemTreeView

from yologait import YOLOGait

from sort_in_yolov5 import *
from datetime import datetime 

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

# 多线程实时检测
class DetectThread(QThread):
    Send_signal = pyqtSignal(np.ndarray, int)
    Output_signal = pyqtSignal(np.ndarray, list)

    def __init__(self, fileName, actul_fps, video_fps):
        super(DetectThread, self).__init__()
        self.capture = cv2.VideoCapture(fileName)
        self.count = 0
        if actul_fps < 1.0:
            self.skip_num = round((float(str(1.00/actul_fps)[:4]) + 0.1) * video_fps)
        else:
            self.skip_num = round((float(str(1.00/actul_fps)[:4])) * video_fps)
        self.warn = False 

    def run(self):
        ret, self.frame = self.capture.read()
        while ret:
            ret, self.frame = self.capture.read()
            if self.count % self.skip_num == 0:
                self.detectCall()
            self.count += 1

    def detectCall(self):
        fps = 0.0
        t1 = time.time()
        frame = self.frame
        frame_new, person_in_danger, person_to_risk_parm, id_in_danger = yologait.detect_image(frame, window.dangerous_parm)
        if person_in_danger:
            self.Output_signal.emit(frame_new, [person_in_danger, person_to_risk_parm, id_in_danger])
        frame = np.array(frame_new)
        fps = 1./(time.time()-t1)
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.Send_signal.emit(frame, self.warn)

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.thread_status = False  # 判断识别线程是否开启
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction("向右旋转90", self)
        self.action_left_rotate = QAction("向左旋转90°", self)
        self.action_opencam = QAction("开启摄像头", self)
        self.action_video = QAction("加载视频", self)
        self.action_image = QAction("加载图片", self)
        self.action_background = QAction("设置危险区域", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.action_opencam.triggered.connect(self.opencam)
        self.action_video.triggered.connect(self.openvideo)
        self.action_image.triggered.connect(self.openimage)
        self.action_background.triggered.connect(self.background)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate,
                                  self.action_opencam, self.action_video, self.action_image, 
                                  self.action_background))
        self.stackedWidget = StackedWidget(self)
        self.graphicsView = GraphicsView(self)
        self.dock_file = QDockWidget(self)
        self.dock_file.setTitleBarWidget(QLabel('参数设置'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_file.setMinimumWidth(200)
        self.dockWidgetContents = QtWidgets.QWidget()
        self.verticalLayoutWidget = QtWidgets.QWidget(self.dockWidgetContents)

        self.param_layout = QVBoxLayout(self.verticalLayoutWidget)
        
        self.get_param1 = QPushButton('设置目标检测模型')
        self.param1 = QLabel('')
        self.get_param1.clicked.connect(self.get_file1)
        
        self.get_param2 = QPushButton('设置轮廓分割模型')
        self.param2 = QLabel('')
        self.get_param2.clicked.connect(self.get_file2)
        
        self.get_param3 = QPushButton('设置步态识别模型: 一阶段')
        self.param3 = QLabel('')
        self.get_param3.clicked.connect(self.get_file3)

        self.get_param7 = QPushButton('设置步态识别模型: 二阶段')
        self.param7 = QLabel('')
        self.get_param7.clicked.connect(self.get_file7)
        
        self.get_param4 = QPushButton('选择步态数据库位置')
        self.get_param4.clicked.connect(self.get_file4)
        self.param4 = QLabel('')
        self.param5 = QLabel('运行位置: cpu or gpu')
        self.get_param5 = QLineEdit()
        self.get_param5.textChanged.connect(self.text_changed5)
        self.param6 = QLabel('视频尺寸: 宽度')
        self.get_param6 = QLineEdit()
        self.get_param6.textChanged.connect(self.text_changed6)
        self.param8 = QLabel('检测帧率: fps')
        self.get_param8 = QLineEdit()
        self.get_param8.textChanged.connect(self.text_changed8)
        self.param9 = QLabel('视频帧率: fps')
        self.get_param9 = QLineEdit()
        self.get_param9.textChanged.connect(self.text_changed9)
        self.get_param10 = QPushButton('系统各模型开始加载')
        self.get_param10.clicked.connect(self.model_load)
        self.param_layout.addWidget(self.get_param1)
        self.param_layout.addWidget(self.param1)
        self.param_layout.addWidget(self.get_param2)
        self.param_layout.addWidget(self.param2)
        self.param_layout.addWidget(self.get_param3)
        self.param_layout.addWidget(self.param3)
        self.param_layout.addWidget(self.get_param7)
        self.param_layout.addWidget(self.param7)
        self.param_layout.addWidget(self.get_param4)
        self.param_layout.addWidget(self.param4)
        self.param_layout.addWidget(self.param5)
        self.param_layout.addWidget(self.get_param5)
        self.param_layout.addWidget(self.param6)
        self.param_layout.addWidget(self.get_param6)
        self.param_layout.addWidget(self.param8)
        self.param_layout.addWidget(self.get_param8)
        self.param_layout.addWidget(self.param9)
        self.param_layout.addWidget(self.get_param9)
        self.param_layout.addWidget(self.get_param10)
        self.dock_file.setWidget(self.dockWidgetContents)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('监测数据'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.setCentralWidget(self.graphicsView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('危险区域侵入行为主动管理系统')
        self.setWindowIcon(QIcon('icons/mask.png'))
        self.resize(1280,720)

        self.src_img = None
        self.cur_img = None
        self.yolo_weight = ''
        self.rvm_weight = ''
        self.gait_weight1 = ''
        self.gait_weight2 = ''
        self.gait_data = ''
        self.device = 'cpu'
        self.imgsz = 1280
        self.dangerous_area = []
        self.dangerous_parm = []
        self.actul_fps = 30
        self.video_fps = 30

    def get_file1(self):
        fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择权重文件")
        self.param1.setText(fileName)
        self.yolo_weight = fileName
    def get_file2(self):
        fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择权重文件")
        self.param2.setText(fileName)
        self.rvm_weight = fileName
    def get_file3(self):
        fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择权重文件")
        self.param3.setText(fileName)
        self.gait_weight1 = fileName
    def get_file7(self):
        fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择权重文件")
        self.param7.setText(fileName)
        self.gait_weight2 = fileName
    def get_file4(self):
        fileName = QFileDialog.getExistingDirectory(
                self, "选择所在文件夹")
        self.param4.setText(fileName)
        self.gait_data = fileName
    def text_changed5(self):
        device = self.get_param5.text()
        print(device)
        self.device = device
    def text_changed6(self):
        imgsz = self.get_param6.text()
        print(imgsz)
        self.imgsz = int(imgsz)
    def text_changed8(self):
        actul_fps = self.get_param8.text()
        print(actul_fps)
        self.actul_fps = float(actul_fps)
    def text_changed9(self):
        video_fps = self.get_param9.text()
        print(video_fps)
        self.video_fps = float(video_fps)

    def model_load(self):
        yologait.generate(self.yolo_weight, self.rvm_weight, self.gait_weight1, self.gait_weight2,
                          self.gait_data, self.device, self.imgsz, self.dangerous_area)
    
    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.change_image(img)

    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    def right_rotate(self):
        self.graphicsView.rotate(90)

    def left_rotate(self):
        self.graphicsView.rotate(-90)

    def add_item(self, image, list_person):
        person_in_danger = list_person[0]
        person_to_risk_parm = list_person[1]
        id_in_danger = list_person[2]
        wight = QWidget()
        layout_main = QHBoxLayout()
        layout_right = QVBoxLayout()
        layout_right_down = QHBoxLayout() 
        layout_right_down.addWidget(QLabel(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        cv2.imwrite(os.path.join(self.save_dir, (str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())) + '.jpg')), image)

        layout_right.addWidget(QLabel('注意！检测到有人处于危险区域:'))
        layout_right.addWidget(QLabel(str(person_in_danger)+' [L: '+ person_to_risk_parm[str(id_in_danger[0])][0] +' S: '+ person_to_risk_parm[str(id_in_danger[0])][1]+ ' C: ' + person_to_risk_parm[str(id_in_danger[0])][2]+ ' D: '+ person_to_risk_parm[str(id_in_danger[0])][3] + ']'))
        layout_right.addLayout(layout_right_down)
        layout_main.addLayout(layout_right)
        wight.setLayout(layout_main) 
        item = QListWidgetItem()
        item.setSizeHint(QSize(450, 100)) 
        self.stackedWidget.addItem(item) 
        self.stackedWidget.setItemWidget(item, wight) 

    def openvideo(self):
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), yologait.save_dir, 'danger_workers')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(self.thread_status)
        if self.thread_status == False:

            fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择视频", "D:/", "*.mp4;;*.flv;;All Files(*)")

            flag = self.cap.open(fileName)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请选择视频文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.detectThread = DetectThread(fileName, self.actul_fps, self.video_fps)
                self.detectThread.Send_signal.connect(self.Display)
                self.detectThread.Output_signal.connect(self.add_item)
                self.detectThread.start()
                self.action_video.setText('关闭视频')
                self.thread_status = True
        elif self.thread_status == True:
            self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.action_video.setText('打开视频')
            self.thread_status = False

    def openimage(self):
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), yologait.save_dir, 'danger_workers')
        if self.thread_status == False:
            fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择图片", "D:/", "*.jpg;;*.png;;All Files(*)")
            if fileName != '':
                src_img = cv2.imread(fileName)
                r_image, person_in_danger = yologait.detect_image(src_img)
                if person_in_danger:
                    self.add_item(r_image, person_in_danger)
                r_image = np.array(r_image)
                r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(
                    r_image.data, r_image.shape[1], r_image.shape[0], QtGui.QImage.Format_RGB888)
                self.graphicsView.set_image(QtGui.QPixmap.fromImage(showImage))
    
    def background(self):
        fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择图片", "D:/", "*.jpg;;*.png;;All Files(*)")
        if fileName != '':
            img = cv2.imread(fileName)
            img = cv2.resize(img, (1280, 720))
            self.dangerous_area = []
            window_title = "Background: Please select the boundary of the hazardous area clockwise or counterclockwise"
            def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                list_xy = []
                if event == cv2.EVENT_LBUTTONDOWN:
                    xy = "%d,%d" % (x, y)
                    list_xy.append(x)
                    list_xy.append(y)
                    print(list_xy)
                    cv2.circle(img, (x, y), 1, (0, 0, 0), thickness=-1)
                    cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
                    cv2.imshow(window_title, img)
                    self.dangerous_area.append(list_xy)
            cv2.namedWindow(window_title)
            cv2.setMouseCallback(window_title, on_EVENT_LBUTTONDOWN)
            cv2.imshow(window_title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.dangerous_parm[0], _ = QInputDialog.getDouble(self, "时间阈值T1", "输入数值:")
            print(self.dangerous_parm[0])
            self.dangerous_parm[1], _  = QInputDialog.getDouble(self, "时间阈值T2", "输入数值:")
            print(self.dangerous_parm[1])
            self.dangerous_parm[2], _  = QInputDialog.getInt(self, "不安全行为风险评价参数L", "输入数值:")
            print(self.dangerous_parm[2])
            self.dangerous_parm[3], _  = QInputDialog.getInt(self, "不安全行为风险评价参数C", "输入数值:")
            print(self.dangerous_parm[3])

    def opencam(self):
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), yologait.save_dir, 'danger_workers')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.thread_status == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.detectThread = DetectThread(self.CAM_NUM, self.actul_fps, self.video_fps)
                self.detectThread.Send_signal.connect(self.Display)
                self.detectThread.start()
                self.action_video.setText('关闭视频')
                self.thread_status = True
        else:
            self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.action_video.setText('打开视频')
            self.thread_status = False

    def Display(self, frame, warn):

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(
            im.data, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB888)
        self.graphicsView.set_image(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning, u"关闭", u"确定退出？")
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.thread_status == True:
                self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            event.accept()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    yologait = YOLOGait()
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
