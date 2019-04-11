# Picasso inverse kinematics type3 : input(x,y,z) output(theta2,theta3)
import random
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import keras.backend as K
from keras.utils import to_categorical

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

def build_model(init = 'glorot_uniform'):              # NN Model
    model = keras.Sequential()
    #model.add(keras.layers.Flatten(data_format=None))
    model.add(keras.layers.Dense(3, input_dim=3, kernel_initializer=init))
    #model.add(keras.layers.Dense(100,use_bias=True, activation='tanh'))
    #model.add(keras.layers.Dense(100, use_bias=True, activation='tanh'))
    model.add(keras.layers.Dense(200, use_bias=True, activation='relu', kernel_initializer=init))    #relu, tanh, softmax, linear, sigmoid
    model.add(keras.layers.Dense(2, use_bias=True, activation='linear', kernel_initializer=init))
    model.compile(optimizer=tf.train.AdamOptimizer(0.05), loss=customloss, metrics=['accuracy'])  # 0.05
    return model
    
# tf.train.AdamOptimizer(0.05)
# sgd                                         随机梯度下降法

# customloss                                  defined by self
# mean_squared_error或mse                     均方误差，也称标准差，缩写为MSE，可以反映一个数据集的离散程度。
# mean_absolute_error或mae                    平均绝对误差，平均绝对误差是所有单个观测值与算术平均值的偏差的绝对值的平均。
# mean_absolute_percentage_error或mape        平均绝对百分比误差
# mean_squared_logarithmic_error或msle        均方对数误差
# squared_hinge                           	  取1减去预测值与实际值的乘积的结果与0比相对大的值的平方的累加均值
# hinge                                       取1减去预测值与实际值的乘积的结果与0比相对大的值的累加均值。
# binary_crossentropy（亦称作对数损失，logloss）  对数损失函数，log loss，与sigmoid相对应的损失函数。
# categorical_crossentropy                    亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
# sparse_categorical_crossentrop              如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：np.expand_dims(y,-1)
# kullback_leibler_divergence                 从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
# cosine_proximity                            即预测值与真实标签的余弦距离平均值的相反数

def plot_history(history):      # Track the history
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label ='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='validation Lost')
    plt.legend()
    plt.ylim([0,10])

class printDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


file = open ("traing_data_for_doyle_type3.csv","w")                             # Data Set Creation
for i in range (0,samples):
    q1= round(random.uniform(0.01, math.pi), 2)
    q2= round(random.uniform(0.01, math.pi), 2)
    q3= round(random.uniform(0.01, math.pi), 2)

    Q2.append(q2)
    file.write(str(q2))
    file.write(",")

    Q3.append(q3)
    file.write(str(q3))
    file.write(",")

    X = Xe(q1,q2,q3)
    posX.append(X)
    file.write(str(round(X, 2)))
    file.write(",")

    Y = Ye(q1,q2,q3)
    posY.append(Y)
    file.write(str(round(Y, 2)))
    file.write(",")

    Z = Ze(q1,q2,q3)
    posZ.append(Z)
    file.write(str(round(Z, 2)))
    file.write("\n")

file.close()

ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
for i in range(0,len(posX)):
    ax.scatter(posX[i], posY[i], posZ[i], c='y')  # 绘制数据点

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Data set of endeffector positions and orientations")
#plt.show( block = False )
plt.savefig('Data set of endeffector positions and orientations for doyle_type3.png')


dataMat =np.c_[Q2,Q3,posX,posY,posZ]          # Augmenting to the data marix

for i in range (0,samples):                         # Removing duplicated end effeector positions
    check1 = dataMat[i,2]
    check2 = dataMat[i,3]
    check3 = dataMat[i,4]
    for j in range (0,samples):
        if i != j:
            #print(i,j,'checking..')
            if (dataMat[j,2] == check1 and dataMat[j,3] == check2 and dataMat[j,4] == check3):
                print(i,j,dataMat[j,2],dataMat[j,3],dataMat[j,4])
                del Q2[j]
                del Q3[j]
                del posX[j]
                del posX[j]
                del posZ[j]
                j = j -1
print("Duplicated end effeector positions have been removed!")    
         
dataMat =np.c_[Q2,Q3,posX,posY,posZ]          # Augmenting to the data marix

data = dataMat[:,[2,3,4]] #X,Y,Z
output = dataMat[:,[0,1]]  # Q2,Q3

train_input = data[0:int(0.7*samples),:]                             #Separate data set in to Train, Test And Validation
train_output = output[0:int(0.7*samples),:]

test_input = data[int(0.7*samples):int(0.85*samples),:]
test_output = output[int(0.7*samples):int(0.85*samples),:]

validate_input = data[int(0.85*samples):int(samples),:]
validate_output = output[int(0.85*samples):int(samples),:]

print("Train INPUT---------------------")
print(np.shape(train_input))
print(train_input)
print("OUTPUT--------------------")
print(np.shape(output))
print(output)

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

NAME = "Trajectry Tracking for doyle_type3"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))           # Create  callbacks for tensorboard visualizations

model = build_model()                                              # Building the model
history = model.fit(dataX, dataY, nb_epoch=300, callbacks=[tensorboard])    #train the model
[loss,mae] = model.evaluate(dataX_test,dataY_test,verbose=0)        #evaluation
print("Testing set Mean Abs Error: ${:7.2f}".format(mae))


dataX_input = x_scaler.transform(validate_input)
test_prediction = model.predict(dataX_input) #predict
real_prediction = y_scaler.inverse_transform(test_prediction)

plt.clf()
plt.scatter(validate_output[:,0],real_prediction[:,0],c='g')              # Plotting Actual angles( x: desired output(Joint angles used to genarate Xe,Ye nd Titae,y: output from prediction )
plt.scatter(validate_output[:,1],real_prediction[:,1],c='r')
plt.xlabel('True Values angles in rad')
plt.ylabel('Predictions  angles in rad')
plt.title("True Value Vs Prediction")
plt.legend("If all predicted values equal to the desired(true) value, this will be lie on 45 degree line")
#plt.show( block = False )

plt.savefig('True Value Vs Prediction for doyle_type3.png')

#_Tensor Board
#tensorboard --logdir=logs/
print("*********************************")
print(validate_input[100,0]," ",validate_input[100,1]," ",validate_input[100,2])
print(Xe(CalQ1(validate_input[100,0],validate_input[100,1]),real_prediction[100,0],real_prediction[100,1])," ",Ye(CalQ1(validate_input[100,0],validate_input[100,1]),real_prediction[100,0],real_prediction[100,1])," ",Ze(CalQ1(validate_input[100,0],validate_input[100,1]),real_prediction[100,0],real_prediction[100,1]))
print("*********************************")

single_data_1 = np.array([[0.5, 0.6, 0.7]])
single_data = x_scaler.transform(single_data_1)
single_prediction = model.predict(single_data)
single_real_prediction = y_scaler.inverse_transform(single_prediction)

print(single_data_1[0,0]," ",single_data_1[0,1]," ",single_data_1[0,2])
print(Xe(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ye(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1])," ",Ze(CalQ1(0.5,0.6),single_real_prediction[0,0],single_real_prediction[0,1]))
print("*********************************")

# serialize model to JSON
model_json = model.to_json()
with open("model_doyle_type3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_doyle_type3.h5")
print("Saved model to disk")
