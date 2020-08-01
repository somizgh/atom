from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import variables
import detect_character_model_variables as DCMV
model = load_model('my_model.h5')

dataX = np.load("./learn_character_model_data/data_X.npy")
dataY = np.load("./learn_character_model_data/data_Y.npy")

scaler=MinMaxScaler()
dataX[:] = scaler.fit_transform(dataX[:])
dataY[:] = scaler.fit_transform(dataY[:])


dataX=dataX.reshape((len(dataX),variables.LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'],variables.LENGTH_OF['RESIZED_IMAGE_VERTICAL'],1))


predict_class = model.predict_classes(dataX)
predict = model.predict(dataX)

errors_num=[]
real_Y=[]
predict_Y = []
predict_Y_output =[]
data_X = []


for i in range(len(predict_class)):
    if dataY[i, predict_class[i]] != 1:
        errors_num.append(i)
        predict_Y.append(predict_class[i])
        predict_Y_output.append(predict[i])
        real_Y.append(dataY[i])

for i in range(len(errors_num)):
    print("font{0:003d}num{1:003d}".format(int(errors_num[i]/variables.NUMBER_OF['TOTAL']), errors_num[i]%variables.NUMBER_OF['TOTAL']), end=" ")
    real_char_code = 0
    for j in range(variables.NUMBER_OF['TOTAL']):
        if real_Y[i][j] == 1:
            real_char_code = j
            break
    print("real_char_code : ",real_char_code,"predict_code : ",predict_Y[i],end=" ")
    print("real char : ",DCMV.code_seq[real_char_code], end=" ")
    print("predict char : ",DCMV.code_seq[predict_Y[i]],end= " ")
    print()

def detect_character():

    return 0

if __name__ == "__main__":

    return 0





