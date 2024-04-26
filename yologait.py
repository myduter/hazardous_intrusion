import os
import cv2
import time

import torch
from pathlib import Path
import sys
from models.common import DetectMultiBackend
from yolov5_utils.general import (letterbox, apply_classifier, check_img_size, increment_path, non_max_suppression, scale_coords)
from yolov5_utils.plots import Annotator, colors
from yolov5_utils.torch_utils import select_device, time_sync
import argparse
import numpy as np

from other_models.rvm import MattingNetwork
from other_models.gaitset.model.initialization import initialization
from other_models.gaitset.model.utils import evaluation
from other_models.gaitset.config import conf as gaitset_conf
from torchvision import transforms

from sort_in_yolov5 import *
from datetime import datetime 

if hasattr(sys, 'frozen'):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

## rvm
class RVM_Net:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.device = device
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.model.eval()
    def convert(self, img, rec):
        img = img.to(self.device, torch.float32, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
        fgr, pha, *rec = self.model(img, *rec, downsample_ratio = 1)
        return pha[0]

class YOLOGait(object):
    def generate(self, yolo_weight, rvm_weight, gait_weight1, gait_weight2,
                          gait_data, device, imgsz, dangerous_area):
        self.device = select_device(device)
        self.dangerous_area = dangerous_area
        self.l = 0
        self.t0 = time.time()
        out, weights, self.imgsz = \
            os.path.join(application_path, 'inference/output'), yolo_weight, imgsz

        # Directories
        self.save_dir = increment_path(Path(out) / 'exp', exist_ok=False)  # increment run
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        if not os.path.exists(os.path.join(self.save_dir, 'backgrounds')):
            os.makedirs(os.path.join(self.save_dir, 'backgrounds'))
        self.record_path = str(self.save_dir / 'safe_record.txt')
        record_file = open(self.record_path, "w", encoding="utf-8")
        record_file.writelines(['工人'+';', '事故发生可能性L'+';', 
                                '工人不安全状态S'+';', '事故后果严重性C'+';', 
                                '工人危险指数D'+';', '开始时间', ';', 
                                '持续时间', ';', '危险区域类型', '\n'])
        record_file.close()
        

        # Load model
        print('Load yolov5 model...')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=os.path.join(application_path, '.yaml'))

        print('Load rvm model...')
        self.rvm_model = RVM_Net(variant='mobilenetv3', device=self.device, checkpoint=rvm_weight)

        print('Load gaitset Model...')
        gaitset_conf['data']["dataset_path"] = gait_data
        gaitset_conf["CUDA_VISIBLE_DEVICES"] = self.device.type
        self.gaitset_m = initialization(gaitset_conf, test=False)[0]
        self.gaitset_m.encoder.load_state_dict(torch.load(gait_weight1, map_location=self.device))
        feature_data = self.gaitset_m.transform('test')
        self.gallery_x, self.gallery_y = evaluation(feature_data, gaitset_conf['data'])
        self.gallery_x = torch.from_numpy(self.gallery_x).permute(1, 0, 2).contiguous().to(self.device)
        self.gaitset_m.encoder.load_state_dict(torch.load(gait_weight2, map_location=self.device))
        
        self.mot_tracker = Sort(max_age=15, min_hits=0, iou_threshold=0.4)
        
        self.id_in_danger = []
        self.id_to_person = {}
        self.person_to_risk_parm = {}
        self.danger_start_time = {}
        self.danger_last_time = {}

    def detect_image(self, img0, dangerous_parm):
        stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(self.imgsz, s=stride)
        self.person_in_danger = set([])

        # Half
        half = False
        half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'
        if pt or jit:
            self.model.model.half() if half else self.model.model.float()

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device) 
        _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None 
        plagtrain = 0
        s = ''

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=32, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)  # .to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = self.model(img, augment=False, visualize=False)

        # nms
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        # Second-stage classifier
        rec = [None] * 4
        im0s_pha = transforms.ToTensor()(img0)
        pha = self.rvm_model.convert(im0s_pha, rec)

        if pred[0].size()[0] > 0:
            pred_copy = pred
            pred, c2, plagtrain, delete_x = apply_classifier(pred, self.gaitset_m.encoder, img, img0, pha, plagtrain,
                                                                 self.gallery_x, self.gallery_y, self.device)
            for z in range(len(delete_x)):
                try:
                    tensor1 = pred_copy[0][delete_x[z]]
                    pred[0] = torch.cat((pred[0], tensor1), dim=0)
                except:
                    continue

        t2 = time_sync()
        t3 = time.time()
        for i, det in enumerate(pred):

            if (t3 - self.t0) >= 10:
                cv2.imwrite(os.path.join(self.save_dir, 'backgrounds', ('background' + str(self.l) + '.jpg')), img0)
                self.l += 1
                if self.l >= 10:
                    self.l = 0
                self.t0 = t3

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(img0, line_width=3, example=str(names), dangerous_area=self.dangerous_area)
            if det is not None and len(det):
                det = det.cpu().numpy()
                det = np.insert(det, 5, det[:, 4], axis=1)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                det[:, :5] = self.mot_tracker.update(det[:, :5])  

                det = torch.from_numpy(det)
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()
                    s = '%g %s, ' % (n, names[int(c)])
                k = 0
                for *xyxy, sort_id, conf, cls in reversed(det):
                    sort_id = int(sort_id)
                    a = len(det) - 1
                    people = ['stranger', 'Tong', 'Qian', 'Sun', 'Zhou', 'Li', 'Tong', 'Wu', 'Zheng', 'Wang', 'Feng']
                    label = people[c2[k]] if c == 2 else names[int(c)]
                    self.id_to_person[str(sort_id)] = label

                    if c == 2:
                        box_xyxy = np.float32([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                        actual_site = [(box_xyxy[2] + box_xyxy[0]) / 2, box_xyxy[3]]
                        pts_dst = np.float32(self.dangerous_area)

                        cross = [0] * len(pts_dst)
                        for j in range(len(pts_dst)):
                            if j < (len(pts_dst) - 1):
                                cross[j] = np.cross(actual_site - pts_dst[j], actual_site - pts_dst[j + 1])
                            else:
                                cross[j] = np.cross(actual_site - pts_dst[j], actual_site - pts_dst[0])
                        if (cross[0] > 0 and cross[1] > 0 and cross[2] > 0 and cross[3] > 0) or (
                                cross[0] < 0 and cross[1] < 0 and cross[2] < 0 and cross[3] < 0):

                            if sort_id not in self.id_in_danger:
                                    self.id_in_danger.append(sort_id)
                                    self.person_in_danger.add(self.id_to_person[str(sort_id)])
                                    self.person_to_risk_parm[str(sort_id)] = [str(dangerous_parm[2]), '_', str(dangerous_parm[3]), "_"]
                                    self.danger_start_time[str(sort_id)] = datetime.now()
                                    self.danger_last_time[str(sort_id)] = 0
                            else:
                                self.danger_last_time[str(sort_id)] = (datetime.now()-self.danger_start_time[str(sort_id)]).seconds
                                self.person_in_danger.add(self.id_to_person[str(sort_id)])
                            if self.danger_last_time[str(sort_id)] < dangerous_parm[0]:
                                risk_state = 'Low risk'
                                risk_s = 3
                            elif (dangerous_parm[0] < self.danger_last_time[str(sort_id)]) and (self.danger_last_time[str(sort_id)] < dangerous_parm[1]):
                                risk_state = 'Medium risk'
                                risk_s = 6
                            else:
                                risk_state = 'High risk'
                                risk_s = 10
                            risk_d = risk_s * dangerous_parm[2] * dangerous_parm[3]
                            if risk_d <= 20:
                                risk_level = 5
                            elif (risk_d > 20) and (risk_d <= 70):
                                risk_level = 4
                            elif (risk_d > 70) and (risk_d <= 160):
                                risk_level = 3
                            elif (risk_d > 160) and (risk_d <= 320):
                                risk_level = 2
                            else:
                                risk_level = 1

                            condition = 'Risk level: ' + str(risk_level) + '. (' + str(48) + ' seconds)'
                            try:
                                self.person_to_risk_parm[str(sort_id)][1] = str(risk_s)
                                self.person_to_risk_parm[str(sort_id)][3] = str(risk_d)
                            except:
                                print(0)
                                condition = None
                                if sort_id in self.id_in_danger:
                                    record_file = open(self.record_path, "a", encoding="utf-8")
                                    record_file.writelines([label+';', self.person_to_risk_parm[str(sort_id)][0]+';', 
                                                            self.person_to_risk_parm[str(sort_id)][1]+';', self.person_to_risk_parm[str(sort_id)][2]+';', 
                                                            self.person_to_risk_parm[str(sort_id)][3]+';', self.danger_start_time[str(sort_id)].strftime('%Y-%m-%d'), ';', 
                                                            str(self.danger_last_time[str(sort_id)]), ';', '物体打击区', '\n'])
                                    record_file.close()
                                    self.id_in_danger.remove(sort_id)
                                    self.person_in_danger.discard(self.id_to_person[str(sort_id)])
                                    del self.person_to_risk_parm[str(sort_id)]
                                    del self.id_to_person[str(sort_id)]
                                    del self.danger_start_time[str(sort_id)]
                                    del self.danger_last_time[str(sort_id)]
                            
                        else:
                            condition = None
                            if sort_id in self.id_in_danger:
                                    record_file = open(self.record_path, "a", encoding="utf-8")
                                    record_file.writelines([label+';', self.person_to_risk_parm[str(sort_id)][0]+';', 
                                                            self.person_to_risk_parm[str(sort_id)][1]+';', self.person_to_risk_parm[str(sort_id)][2]+';', 
                                                            self.person_to_risk_parm[str(sort_id)][3]+';', self.danger_start_time[str(sort_id)].strftime('%Y-%m-%d'), ';', 
                                                            str(self.danger_last_time[str(sort_id)]), ';', '物体打击区', '\n'])
                                    record_file.close()
                                    self.id_in_danger.remove(sort_id)
                                    self.person_in_danger.discard(self.id_to_person[str(sort_id)])
                                    del self.person_to_risk_parm[str(sort_id)]
                                    del self.id_to_person[str(sort_id)]
                                    del self.danger_start_time[str(sort_id)]
                                    del self.danger_last_time[str(sort_id)]
                    else:
                        condition = None
                    c = int(cls)
                    annotator.box_label(xyxy, label, condition, color=colors(c, True))
                    a = a - 1
                    k = k + 1

            print('%sDone. (%.3fs)' % (s, t2 - t1))

        return img0, self.person_in_danger, self.person_to_risk_parm, self.id_in_danger


