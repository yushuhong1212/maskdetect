import matplotlib.pyplot as plt
from mtcnn import MTCNN
import cv2 as cv
import tensorflow as tf
import time
import os #导入库

model = tf.keras.models.load_model("C:/Users/123/Tensorf/model.h5")#读取模型


#打开摄像头
cap=cv.VideoCapture(0)
#初始化检测器
detector=MTCNN()

def detectTime():
    while cap.isOpened():
        OK,frame=cap.read()
        start_time=time.time()
        #检测图片摄像头或者视频中的人脸
        results=detector.detect_faces(frame)
        for bound in results:
            #boound['box']包含四个值:x,y,w,h
            bound_box=bound['box']
            confidence='%.3f'%(bound['confidence'])
            x0,y0,w,h=bound_box[0],bound_box[1],bound_box[2],bound_box[3]
            x1=x0+w
            y1=y0+h
            image = frame[(y0-10):(y1+10), (x0-10):(x1+10)]
            img1 = cv.resize(image,(128,128,))
            img1 = img1.reshape((-1,128,128,3))
            pre = model.predict(img1)
            pre = np.argmax(pre)
            if pre==0:
                cv.rectangle(img=frame,pt1=(x0,y0),pt2=(x1,y1),color=(0, 255, 0),thickness=2)
                cv.putText(img=frame,text='WithMask'+confidence,org=(x0,y0-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.2,color=(0,255,0),thickness=2)
            else:
                cv.rectangle(img=frame,pt1=(x0,y0),pt2=(x1,y1),color=(0, 0, 255),thickness=2)
                cv.putText(img=frame,text='NoMask',org=(x0,y0-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.2,color=(0,0,255),thickness=2)
        end_time=time.time()
        #计算出帧数
        FPS=1/(end_time-start_time)
        cv.putText(frame, 'FPS ' + str(int(FPS)), (20, 50 ), cv.FONT_HERSHEY_SIMPLEX, 1.25,
                          (200, 200, 200), 1)
        cv.imshow('face',frame)
        if cv.waitKey(1) & 0XFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    print('Pycharm')
    detectTime()