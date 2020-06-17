"""
학습된 모델을 서비스로 활용하기 위해 로더 사용
"""

# Sklearn pkl 모델 로더
# Sklearn 의 경우 모델에 대한 처리 방식이 Tensorflow/Keras 와 달라
# 아래 클래스를 상속받아 Custom Class 구현이 필요
from aicentro.loader.sklearn_loader import SklearnLoader

class CustomLoader(SklearnLoader):

    def __init__(self, models=None):
        """
        init 함수이며 추가로 작성할 내용이 있으면 작성
        :param models: 모델파일명 리스트
        """
        super().__init__(models)

    def predict(self, inputs_key=None, inputs=None):
        """
        실제 로드된 모델을 가지고 출력값을 만들어 내는 함수
        :param inputs_key: 사용 안함
        :param inputs: 입력값
        :return: 출력값
        """
        # self.models['모델명'] 으로 사용할 모델 정보를 호출
        # 해당 내용에서 predict 또는 predict_proba 등을 사용함
        # 여러 모델이 사용될 경우 순차 실행 순서에 따라 코드 구성

loader = CustomLoader(models=[])

# Base 로더
# 기본적인 로더 클래스이며 필요에 따라 해당 클래스를 상속 받아 구현

from aicentro.loader.base_loader import BaseLoader

class CustomLoader(BaseLoader):

    def __init__(self):
        """
        init 함수이며 추가로 작성할 내용이 있으면 작성
        """
        super().__init__()

    def saved_model_dir(self, version=None):
        """
        모델에 따라 실제 모델 저장 위치를 제공
        :param version:
        :return:
        """
        return self.model_dir

    def print_model_metadata(self):
        """
        모델에 대한 상세 스펙 정보 출력용
        :return:
        """
        return 'spec'

    def predict(self, inputs_key=None, inputs=[]):
        """
        모델 결과 생성 함수
        :param inputs_key:
        :param inputs:
        :return:
        """
        ...


loader = CustomLoader()

"""
모델 온라인 서비스를 위한 BaseServing 클래스
"""
from flask import jsonify, request
from aicentro.serving.base_serving import BaseServing

class CustomServing(BaseServing):
    """
    BaseServing 클래스를 기반으로 전처리/후처리 영역을 구성
    """

    def __init__(self, loader, inputs_key='inputs', outputs=None):
        """
        Serving 클래스 초기화
        :param loader: 모델 로더
        :param inputs_key: 입력 텐서의 키 값
               ( 텐서플로우 .pb 모델의 경우 Metagraph 의 inputs 키 값이 필요함, 필수 )
        :param outputs: 출력 텐서의 키 ( optional )
        """
        super().__init__(loader=loader, inputs_key=inputs_key, outputs=outputs)
        ...

    def pre_processing(self):
        """
        Request Body 의 Json Object 값에 대한 전처리 로직 적용
        self.inputs 변수에 변환된 입력 값을 할당
        """
        _json = request.get_json(silent=True)
        self.inputs = _json[self.inputs_key]

    def post_processing(self, response):
        """
        모델 결과를 받아 최종 출력 포멧 변경의 후처리 로직 적용
        response 객체의 타입은 loader.predict 함수에서 리턴되는 타입과 동일하므로
        그에 맞춰 결과 데이터를 재구성해야 함
        :param response: loader.predict 결과 값
        :return: dict: Json 으로 생성될 Dictionary 객체
        """
        resp_dict = dict()
        resp_dict['KEY'] = response
        resp_dict['rsp_code'] = '00000'
        resp_dict['rsp_message'] = 'ok'
        return resp_dict

    def get(self):
        """
        선택사항이며 GET 방식으로 전송 시 제공할 내용을 작성
        GET 방식은 pre_processing 과 post_processing 함수를 수행하지 않음
        구현하지 않으면 BaseServing 의 get 함수 호출
        :return: dict: Json 으로 생성될 Dictionary 객체
        """
        ...

    def post(self):
        """
        선택사항이며 POST 방식으로 전송 시 처리할 내용을 작성
        POST 방식은 pre_processing 과 post_processing 함수를 수행함
        구현하지 않으면 BaseServing 의 post 함수 호출
        :return: Any: 타입은 없으나 BaseServing 일 경우는 loader.predict 함수의 리턴타입으로 결정됨
        """
        ...

# 서빙 클래스는 Flask 의 MethodView 방식을 활용하여 구성됨에 따라
# 클래스 객체 생성과는 다른 as_view 라는 함수를 통해 객체 생성
serving = CustomServing.as_view(
    'serving', # view 이름
    loader, inputs_key='inputs' # CustomServing 클래스의 __init__ 함수의 파라메터
)


"""
모델로더와 서빙 클래스를 Flask 프레임워크에 적용 
"""
from flask import Flask, jsonify, request, Response
from aicentro.serving.serving_config import configure_error_handlers

# Flask 객체 생성
app = Flask(__name__)
# Flask 객체에 URL Rule 정의
app.add_url_rule(
    '/', # URL 패스 정의
    view_func=serving, # 서빙 클래스 객체
    methods=['GET', 'POST'] # 서비스 가능 HTTP 메소드
)
# Error Handler 적용
# 에러에 따른 결과 포멧 변경 시 별도 함수 호출
def message_format(code, message):
    """
    에러 발생 시 리턴되는 Json 포멧 정의
    :param code: configure_error_hanlders 함수 호출 시 지정된 code 값
    :param message: error 발생 시 exception 의 메시지
    :return: dict: Json 으로 생성될 Dictionary
    """
    return {
        'error_code': code,
        'error_message': message
    }


configure_error_handlers(app=app, code='99', msg_fn=message_format)

if __name__ == '__main__':
    app.run(host='0.0.0.0')


### 파일명은 app.py 로 변경하여 사용 ###
