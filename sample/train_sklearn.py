from __future__ import print_function
import numpy as np
import pandas as pd

"""
AI 포탈과 연계를 위한 기본 객체 생성
"""
from aicentro.session import Session
session = Session(verify=False)

"""
학습 모델 개발 시 프레임워크별 객체 사용
"""
from aicentro.framework.framework import BaseFramework as Frm
from aicentro.framework.framework import AutomlFramework
framework = Frm(session=session)


"""
모델 코드 작성
"""
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data = load_iris()

X = data.data
y = data.target
y_label = data.target_names.tolist()


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)

logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
y_test_pred = logreg.predict(X_test)
y_test_c = y_test.reshape(-1, 1)
y_test_pred_c = y_test_pred.reshape(-1, 1)

y_test_onehot = pd.get_dummies(y_test).values
y_test_pred_onehot = pd.get_dummies(y_test_pred).values

"""
모델 학습 후 결과를 저장하고 해당 결과를 UI 상에 노출
"""
arc = AutomlFramework()
arc.make_multiLabel_roc_curve(y_test_onehot, y_test_pred_onehot, 'input_name', y_label)
arc.make_confusion_matrix(y_test_onehot, y_test_pred_onehot, 'input_name', y_label)


#framework.plot_confusion_matrix(y_test_c, y_test_pred_c, target_names=y_label, title='Confusion Matrix')
framework.classification_report(y_test_c, y_test_pred_c, target_names=y_label)
#framework.plot_roc_curve(y_test_onehot, y_test_pred_onehot, len(y_label), y_label)

framework.save_joblib(logreg, name='iris_logisticregresion')

