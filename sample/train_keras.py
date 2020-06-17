from __future__ import print_function
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


"""
SACP AI 포탈과 연계를 위한 기본 객체 생성
"""
from aicentro.session import Session
session = Session(verify=False)

"""
학습 모델 개발 시 프레임워크별 객체 사용
"""
from aicentro.framework.keras import Keras as Frm
framework = Frm(session=session)


"""
모델 코드 작성
"""

batch_size = 128
num_classes = 3
epochs = 100

data = load_iris()

X = data.data
y = data.target
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values


X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=1)

model = Sequential()
model.add(Dense(64,input_shape=(4,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.summary()
"""
모델 학습 시 Accuracy 와 Loss 값을 SACP AI 포탈로 전송하여 UI 상에 노출
제공된 Metric Callback 사용 
"""
history = framework.get_metric_callback()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[history])

y_test_pred = model.predict(X_test, batch_size=128, verbose=1)
y_label = data.target_names.tolist()

y_test_c = np.argmax(y_test, axis=1).reshape(-1, 1)
y_test_pred_c = np.argmax(y_test_pred, axis=1).reshape(-1, 1)

"""
모델 학습 후 결과를 저장하고 해당 결과를 UI 상에 노출
"""
framework.plot_confusion_matrix(y_test_c, y_test_pred_c, target_names=y_label, title='Confusion Matrix')
framework.classification_report(y_test_c, y_test_pred_c, target_names=y_label)
framework.plot_roc_curve(y_test, y_test_pred, len(y_label), y_label)


'''
학습된 모델 저장
'''
framework.save_model(model, model_name='iris-classification')
