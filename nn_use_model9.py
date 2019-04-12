import h5py
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.initializers import glorot_uniform
import keras.backend as K
from sklearn import preprocessing

Q2 = []
Q3 = []
posX = []
posY = []
posZ = []
samples = 10000


def customloss (yTrue,yPred):    #error function
    return K.sum((yTrue - yPred)**2)
    
def Xe (a,b,c):                 # return the X,Y,Z for a given 3 joint angles
    return math.cos(a)*math.sin(b) + (math.cos(a)*math.cos(b)*math.sin(c))/2 + (math.cos(a)*math.cos(c)*math.sin(b))/2
def Ye (e,f,g):
    return math.sin(e)*math.sin(f) + (math.cos(f)*math.sin(e)*math.sin(g))/2 + (math.cos(g)*math.sin(e)*math.sin(f))/2
def Ze (h,i,j):
    return math.cos(i) + (math.cos(i)*math.cos(j))/2 - (math.sin(i)*math.sin(j))/2 + 1
    
def CalQ1 (px,py):
    if px == 0 and py == 0 :
        return 0
    if px == 0 and py > 0 :
        return math.atan2(py,px)
    if px == 0 and py < 0 :
        return -math.atan2(py,px)
    if px > 0 and py == 0 :
        return math.atan2(py,px)
    if px < 0 and py == 0 :
        return math.atan2(py,px)
        
    if px > 0 and py > 0 :
        return math.atan2(py,px)
    if px < 0 and py > 0 :
        return math.atan2(py,px)
    if px < 0 and py < 0 :
        return -math.atan2(py,px)
    if px > 0 and py < 0 :
        return -math.atan2(py,px)
        
def train_data(x,y,z):
    single_data_1 = np.array([[x, y, z]])
    single_data = x_scaler.transform(single_data_1)
    single_prediction = loaded_model.predict(single_data)
    single_real_prediction = y_scaler.inverse_transform(single_prediction)

    print(single_data_1[0,0]," ",single_data_1[0,1]," ",single_data_1[0,2])
    print(Xe(CalQ1(x,y),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ye(CalQ1(x,y),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ze(CalQ1(x,y),single_real_prediction[0,0],single_real_prediction[0,1]))
    print("*********************************")

def train_angle(x,y,z):
    single_data_1 = np.array([[x, y, z]])
    single_data = x_scaler.transform(single_data_1)
    single_prediction = loaded_model.predict(single_data)
    single_real_prediction = y_scaler.inverse_transform(single_prediction)

    print(CalQ1(x,y),single_real_prediction[0,0],single_real_prediction[0,1])
    print("****************train data*****************")
    return (CalQ1(x,y),single_real_prediction[0,0],single_real_prediction[0,1])

def jacobian_angle(x,y,z,Q1,Q2,Q3):
    for i in range(1,10) :
    QCurrent = np.matrix([[Q1],
                        [Q2],
                        [Q3]])
    t1 = QCurrent[0,0]
    t2 = QCurrent[1,0]
    t3 = QCurrent[2,0]
    #print(t1,t2,t3)

    xCurrent = math.cos(t1)*math.sin(t2)+(math.cos(t1)*math.cos(t2)*math.sin(t3))/2+(math.cos(t1)*math.cos(t3)*math.sin(t2))/2
    yCurrent = math.sin(t1)*math.sin(t2)+(math.cos(t2)*math.sin(t1)*math.sin(t3))/2+(math.cos(t3)*math.sin(t1)*math.sin(t2))/2
    zCurrent = math.cos(t2)+(math.cos(t2)*math.cos(t3))/2-(math.sin(t2)*math.sin(t3))/2+1
    #print(xCurrent,yCurrent,zCurrent)
	
    deltX = x - xCurrent
    deltY = y - yCurrent
    deltZ = z - zCurrent
    #print(deltX,deltY,deltZ)

    deltPosition = np.matrix([[deltX],
                             [deltY],
                             [deltZ]])

    jaco = np.matrix([[-math.sin(t1)*math.sin(t2)-(math.cos(t2)*math.sin(t1)*math.sin(t3))/2-(math.cos(t3)*math.sin(t1)*math.sin(t2))/2, math.cos(t1)*math.cos(t2)-(math.cos(t1)*math.sin(t2)*math.sin(t3))/2+(math.cos(t1)*math.cos(t2)*math.cos(t3))/2, (math.cos(t1)*math.cos(t2)*math.cos(t3))/2-(math.cos(t1)*math.sin(t2)*math.sin(t3))/2],
                     [math.cos(t1)*math.sin(t2)+(math.cos(t1)*math.cos(t2)*math.sin(t3))/2+(math.cos(t1)*math.cos(t3)*math.sin(t2))/2, math.cos(t2)*math.sin(t1)+(math.cos(t2)*math.cos(t3)*math.sin(t1))/2-(math.sin(t1)*math.sin(t2)*math.sin(t3))/2, (math.cos(t2)*math.cos(t3)*math.sin(t1))/2-(math.sin(t1)*math.sin(t2)*math.sin(t3))/2],
                     [0, -math.sin(t2)-(math.cos(t2)*math.sin(t3))/2-(math.cos(t3)*math.sin(t2))/2, -(math.cos(t2)*math.sin(t3))/2-(math.cos(t3)*math.sin(t2))/2]])
    #print(jaco[0,0],jaco[0,1],jaco[0,2])
    #print(jaco[1,0],jaco[1,1],jaco[1,2])
    #print(jaco[2,0],jaco[2,1],jaco[2,2])
    
    #print(np.linalg.pinv(jaco)[0,0],np.linalg.pinv(jaco)[0,1],np.linalg.pinv(jaco)[0,2])
    #print(np.linalg.pinv(jaco)[1,0],np.linalg.pinv(jaco)[1,1],np.linalg.pinv(jaco)[1,2])
    #print(np.linalg.pinv(jaco)[2,0],np.linalg.pinv(jaco)[2,1],np.linalg.pinv(jaco)[2,2])
    
    deltQ = np.linalg.pinv(jaco)*deltPosition
    #print(deltQ[0,0],deltQ[1,0],deltQ[2,0])
    
    QCurrent = QCurrent + deltQ

    while QCurrent[0,0] > 2*math.pi :
        QCurrent[0,0] = QCurrent[0,0] - 2*math.pi;
    while QCurrent[0,0] < 0 :
        QCurrent[0,0] = QCurrent[0,0] + 2*math.pi;

    while QCurrent[1,0] > 2*math.pi :
        QCurrent[1,0] = QCurrent[1,0] - 2*math.pi;
    while QCurrent[1,0] < 0 :
        QCurrent[1,0] = QCurrent[1,0] + 2*math.pi;

    while QCurrent[2,0] > 2*math.pi :
        QCurrent[2,0] = QCurrent[2,0] - 2*math.pi;
    while QCurrent[2,0] < 0 :
        QCurrent[2,0] = QCurrent[2,0] + 2*math.pi;

    print(QCurrent[0,0], QCurrent[1,0], QCurrent[2,0])
    print("****************jacobian inverse data*****************")
    return (QCurrent[0,0], QCurrent[1,0], QCurrent[2,0])


# load json and create model
json_file = open('model_doyle_type3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json, custom_objects={'GlorotUniform': glorot_uniform()})

# load weights into new model
loaded_model.load_weights("model_doyle_type3.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(0.05), loss=customloss, metrics=['accuracy'])

tmp = np.loadtxt("traing_data_for_doyle_type3.csv", delimiter=",")
tmp1=np.array(tmp)
Q2 = tmp1[0:, 0] 
Q3 = tmp1[0:, 1]  
posX = tmp1[0:, 2]  
posY = tmp1[0:, 3]  
posZ = tmp1[0:, 4]  
pass #调试点，导入了data为数据，label为标签

dataMat =np.c_[Q2,Q3,posX,posY,posZ]          # Augmenting to the data marix

data = dataMat[:,[2,3,4]] #X,Y,Z
output = dataMat[:,[0,1]]  # Q2,Q3

train_input = data[0:int(0.7*samples),:]                             #Separate data set in to Train, Test And Validation
train_output = output[0:int(0.7*samples),:]

test_input = data[int(0.7*samples):int(0.85*samples),:]
test_output = output[int(0.7*samples):int(0.85*samples),:]

validate_input = data[int(0.85*samples):int(samples),:]
validate_output = output[int(0.85*samples):int(samples),:]

x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))

dataX = x_scaler.fit_transform(train_input)
dataY = y_scaler.fit_transform(train_output)
dataX_test = x_scaler_test.fit_transform(test_input)
dataY_test = y_scaler_test.fit_transform(test_output)
dataX_eva = x_scaler_eva.fit_transform(validate_input)
dataY_eva = y_scaler_eva.fit_transform(validate_output)

single_data_1 = np.array([[0.5, 0.6, 0.7]])
single_data = x_scaler.transform(single_data_1)
single_prediction = loaded_model.predict(single_data)
single_real_prediction = y_scaler.inverse_transform(single_prediction)

print(single_data_1[0,0]," ",single_data_1[0,1]," ",single_data_1[0,2])
print(Xe(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ye(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ze(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1]))
print("*********************************")

train_angle(0.5,0.6,0.7)
jacobian_angle(0.5,0.6,0.7,1.5,1.5,1.5)
