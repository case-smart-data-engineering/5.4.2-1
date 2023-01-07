import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import sys
sys.path.append(os.getcwd())
from line_profiler import LineProfiler
from tqdm import tqdm
import matplotlib.pyplot as plt
from siamfc import SiamFCTracker
from PIL import Image

a = []
def main(video_dir, gpu_id, model_path):

    # 加载视频（图片组）
    filenames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")),key=lambda x: int(os.path.basename(x).split('.')[0]))
    #print(filenames)
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    print(frames[1].shape)

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = SiamFCTracker(model_path, gpu_id)
    t1 = time.time()
    result = []
    for idx, frame in enumerate(frames):
        if idx == 0:
            f =  open('./video/GOT-10k_Test_000009/site.txt',encoding='UTF-8')
            con = f.read()
            f.close()
            bbox = []
            for index in range(0,len(con.split(','))):
                #print(len(con.split(',')))
                bbox.append(int(con.split(',')[index]))
            print(bbox)
            bbox_ori = bbox
            bbox_xy = bbox  #左上+右下坐标格式
            bbox_wh = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                       )  #左上坐标，与宽高
            # 将第一帧，以及第一帧的bbox作为输入
            tracker.init(frame, bbox_wh)
        else:
            # 除了第一帧外的其他帧，找到该帧上兴趣区域的位置


            bbox_xy = tracker.update(frame)
            #测试时间损耗

            # lp = LineProfiler()
            # lp_wrapper = lp(tracker.update)
            # lp_wrapper(frame,idx)
            # lp.print_stats()
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(

            frame,
            (int(bbox_xy[0]), int(bbox_xy[1])),
            (int(bbox_xy[2]), int(bbox_xy[3])),
            (0, 255, 0),  #green
            2)
        frame = cv2.circle(frame, (int((bbox_xy[0] + bbox_xy[2]) / 2), int((bbox_xy[1] + bbox_xy[3]) / 2)),
                           2, (0, 0, 255),  # green
                           2)
        # print("x:", int((bbox_xy[0] + bbox_xy[2]) / 2), "y:", int((bbox_xy[1] + bbox_xy[3]) / 2))
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result.append(cv2.putText(frame, str(idx), (5, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1))
        #plt.imshow(frame)
    for index in range(0,len(result)):
        plt.imshow(result[index])
        plt.show()
        image = result[index]
        name = "test_result_"+str(index)+".jpg"
        #image.save(name)
        cv2.imwrite(name,image)    
        
if __name__ == "__main__":
    # 测试视频根目录
    video_dir = r'./video/GOT-10k_Test_000009'
    # 模型地址
    gpu_id = 0
    model_path = './model/siamfc_pretrained.pth'

    main(video_dir, gpu_id, model_path)
