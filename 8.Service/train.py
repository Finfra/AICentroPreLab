# SACP AI 포탈과 연계를 위한 기본 객체 생성

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from aicentro.session import Session
session = Session(verify=False)

# AI 포탈과 연계를 위한 기본 객체 생성

from aicentro.session import Session
session = Session(verify=False)
from aicentro.framework.framework import AutomlFramework

# 학습 모델 개발 시 프레임워크별 객체 사용

from aicentro.framework.keras import Keras as SacpFrm
framework = SacpFrm(session=session)

# 모델 코드 작성

batch_size = 50
num_classes = 3
epochs = 200

import pandas as pd
iris=pd.read_csv("/data/iris.csv")
iris.drop(['no'],1,inplace=True)
iriss=iris.sample(frac=1).reset_index(drop=True)
iriss.head()

# Shuffling
iriss=iris.sample(frac=1).reset_index(drop=True)
iris_train=iriss.iloc[0:100,:]
iris_test=iriss.iloc[100:150,:]

x_train=iris_train.iloc[:,0:4].values
x_test=iris_test.iloc[:,0:4].values
y_train=iris_train.iloc[:,4:5]
y_test= iris_test.iloc[:,4:5]
# encoder={k:v for v,k in enumerate(y_train.drop_duplicates())}
# encoder
sets=iris.iloc[:,4:5].drop_duplicates()["Species"].tolist()
encoder={k:v for v,k in enumerate(sets)}
y_train=[ encoder[i] for i in y_train["Species"].tolist() ]
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test=[ encoder[i] for i in y_test["Species"].tolist() ]
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(4,)))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 모델 학습 시 Accuracy 와 Loss 값을 SACP AI 포탈로 전송하여 UI 상에 노출
제공된 Metric Callback 사용 

history = framework.get_metric_callback()

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[history])


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 모델 학습 후 결과를 저장하고 해당 결과를 UI 상에 노출


#model.predict(x_test)
y_test_pred = model.predict(x_test, batch_size=128, verbose=1)

y_label = sets

y_test_c = np.argmax(y_test, axis=1).reshape(-1, 1)
y_test_pred_c = np.argmax(y_test_pred, axis=1).reshape(-1, 1)

arc = AutomlFramework()
arc.make_expert_acc_loss_chart(history.metrics)
arc.make_multiLabel_roc_curve(y_test_pred, y_test, 'input_name', y_label)
arc.make_confusion_matrix(y_test_pred, y_test, 'input_name', y_label)
framework.classification_report(y_test_c, y_test_pred_c, target_names=y_label)


framework.plot_confusion_matrix(y_test_c, y_test_pred_c, target_names=y_label, title='Confusion Matrix')


framework.classification_report(y_test_c, y_test_pred_c, target_names=y_label)

framework.plot_roc_curve(y_test, y_test_pred, len(y_label), y_label)


# 학습된 모델 저장

framework.save_model(model, model_name='iris-classification-study')
