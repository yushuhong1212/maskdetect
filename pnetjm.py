
import numpy as np

#-----------------------------#
#   将长方形调整为正方形
#-----------------------------#
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles

#对Pnet处理后的结果进行处理
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, thresold):
    #计算特征点之间的步长
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
        
    #获得满足得分门限的特征点得分
    score = np.expand_dims(cls_prob[y,x], -1)
    
    #将对应的特征点的坐标转换成位于原图上的先验框的坐标
    bounding_box = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1), axis = -1])
    top_left = np.fix(stride * bounding_box + 0 )
    bottom_right = np.fix(stride * bounding_box + 11 )
    bounding_box = np.concatenate((top_left, bottom_right), axis = -1)
    bounding_box = (bounding_box + roi[y,x] * 12.0) * scale
    
    #将预测框和得分堆叠，转换成正方形
    rectangles = np.concatenate((bounding_box, score), axis = -1)
    rectangles = rect2square(rectangles)
    
    retangles[:, [1,3]] = np.clip(retangles[:,[1, 3]], 0, height)
    retangles[:, [0,2]] = np.clip(retangles[:,[0, 2]], 0, width)
    return rectangles