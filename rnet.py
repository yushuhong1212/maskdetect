
#-------------------------------------#
#   非极大抑制
#-------------------------------------#
def NMS(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

#精修框
def create_Rnet(weight_path):
    inputs = tf.keras.Input(shape=[24,24,3])
    #24，24，3 -> 22,22,28 ->11,11,28
    x = tf.keras.layers.Conv2D(28,(3,3), strides=1, padding='valid', name='conv1')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1,2], name = 'PReLU1')(x)
    x = tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same')(x)
    
    #11,11,28 -> 9,9,48 ->4,4,48
    x = tf.keras.layers.Conv2D(48, (3,3), strides = 1,padding = 'valid', name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes = [1,2], name = 'PReLU2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2)(x)

    #4,4,48 -> 3,3,64
    x = tf.keras.layers.Conv2D(64,(2,2),strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2], name='PReLU3')(x)
    
    #3,3,64 -> 64,3,3
    x = tf.keras.layers.Permute((3,2,1))(x)
    x = tf.keras.layers.Flatten()(x)
    
    #576 -> 128
    x = tf.keras.layers.Dense(128, name = 'conv4')(x)
    x = tf.keras.layers.PReLU(name = 'PReLU4')(x)
    
    #128 -> 2
    classifier = tf.keras.layers.Dense(2, activation='softmax', name = 'conv5-1')(x)
    #128 -> 4
    bbox_regress = tf.keras.layers.Dense(4, name='conv5-2')(x)
    
    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model