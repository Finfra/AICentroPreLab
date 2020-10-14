import pandas as pd

"""
SACP AI 포탈과 연계를 위한 기본 객체 생성
"""
from aicentro.session import Session
sacp_session = Session(verify=False)


"""
학습 모델 개발 시 프레임워크별 객체 사용

:type 텐서플로우: from aicentro.framework.tensorflow import Tensorflow as SacpFrm
:type 케라스: from aicentro.framework.keras import Keras as SacpFrm
:type 그 외(Sklearn 등): from aicentro.framework.framework import BaseFramework as SacpFrm

기본 사용법 : SacpFrm(session=세션변수)

"""
from aicentro.framework.keras import Keras as SacpFrm
sacp_framework = SacpFrm(session=sacp_session)

"""
SacpFrm 객체를 활용한 SACP 디렉토리 정보 구하기

아래 변수에 적용되는 값은 분석 IDE, 모델학습, 모델서비스 단계별로 
다른 위치를 바라보게 적용됩니다. 
따라서 변수로 처리 시에는 단계별로 소스 코드의 수정 없이 수행을 할 수 있습니다.
"""

input_data = pd.read_csv(sacp_framework.config.data_dir + '/input.csv')


"""
모델 코드 작성
"""

history = sacp_framework.get_metric_callback()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])


"""
모델 학습 후 결과를 저장하고 해당 결과를 UI 상에 노출
"""
# Confusion Matrix
sacp_framework.plot_confusion_matrix(
    y_true=y_true, # Label(Y) 의 정답 (numpy.array)
    y_pred=y_pred, # Label(Y) 의 예측결과 (numpy.array)
    target_names=y_label, # Label(Y) 의 이름 (array)
    title='Confusion Matrix', # matplotlib 객체의 타이틀 (옵션)
    cmap=None, # matplotlib 의 컬러맵 plt.get_cmap('Greys') ( 기본값 : None )
    normalize=False # 결과 값 Normalize 여부 ( 기본값 : True )
)

# ROC Curve
# 단 y_true , y_pred 는 n_classes 갯수 만큼 One-Hot 인코딩 된 상태의 2 차원 이상으로 구성되어야 함
sacp_framework.plot_roc_curve(
    y_true=y_true, # Label(Y) 의 정답 (numpy.array)
    y_pred=y_pred, # Label(Y) 의 예측결과 (numpy.array)
    n_classes=len(y_label), # Label(Y) 의 이름 갯수
    target_names=y_label, # Label(Y) 의 이름 (array)
    title='ROC Curve' # matplotlib 객체의 타이틀 (옵션)
)

# Classification Report
sacp_framework.classification_report(
    y_true=y_true, # Label(Y) 의 정답 (numpy.array)
    y_pred=y_pred, # Label(Y) 의 예측결과 (numpy.array)
    target_names=y_label # Label(Y) 의 이름 (array)
)

# 추가 결과 이미지 생성
...
plt.figure(figsize=(8, 6), dpi=150) # 화면 배치 기준 50%
plt.savefig(
    sacp_framework.get_plot_save_filename(), # 자동으로 순차적인 파일명을 리턴
    dpi=150,
    bbox_inches='tight',
    pad_inches=0.5
)


"""
학습된 모델을 저장하여 서비스로 활용
"""
# Keras 모델을 .h5 로 저장
sacp_framework.save_model(
    model=model, # Keras 모델 객체
    model_name='ModelName', # 저장 시 사용되는 파일명
    weight=False # Weight 값 분리 여부 ( True 일 경우 weight json 파일 추가 생성 )
)

# Tensorflow/Keras 모델 사용 시 필요한 오브젝트 저장
# 또는 sklearn 으로 학습된 모델 저장
sacp_framework.save_pickle(
    obj=obj, # 저장할 Object 객체
    name='labelNames' # 저장 시 사용되는 파일명
)

