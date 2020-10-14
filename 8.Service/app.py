import numpy as np

"""
학습된 모델을 서비스로 활용하기 위해 로더 사용
"""

# Keras .h5 모델 로더
from aicentro.loader.keras_loader import KerasLoader

loader = KerasLoader(
    model_filename='iris-classification-study'  # 저장된 모델파일명 ( .h5 제외 )
)

"""
모델 온라인 서비스를 위한 BaseServing 클래스
"""
from flask import jsonify, request
from flask import current_app
from aicentro.serving.base_serving import BaseServing


class CustomServing(BaseServing):
    """
    BaseServing 클래스를 기반으로 전처리/후처리 영역을 구성
    """

    def __init__(self, loader, inputs_key='inputs', outputs=None, labels=[]):
        """
        Serving 클래스 초기화
        :param loader: 모델 로더
        :param inputs_key: 입력 텐서의 키 값
               ( 텐서플로우 .pb 모델의 경우 Metagraph 의 inputs 키 값이 필요함, 필수 )
        :param outputs: 출력 텐서의 키 ( optional )
        """
        super().__init__(loader=loader, inputs_key=inputs_key, outputs=outputs)
        self.labels = labels

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
        resp_dict['classification'] = self.labels[np.argmax(response.reshape(-1))]

        current_app.logger.info('response : ' + resp_dict['classification'])

        return resp_dict


# 서빙 클래스는 Flask 의 MethodView 방식을 활용하여 구성됨에 따라
# 클래스 객체 생성과는 다른 as_view 라는 함수를 통해 객체 생성
serving = CustomServing.as_view(
    'serving',  # view 이름
    loader, inputs_key='inputs',  # CustomServing 클래스의 __init__ 함수의 파라메터
    labels=['setosa', 'versicolor', 'virginica']
)

"""
모델로더와 서빙 클래스를 Flask 프레임워크에 적용 
"""
import os
from flask import Flask, jsonify, request, Response
from aicentro.serving.serving_config import configure_error_handlers

context = os.getenv('AICENTRO_NOTEBOOK_CONTEXT_PATH', default='')

# Flask 객체 생성
app = Flask(__name__)
# Flask 객체에 URL Rule 정의
app.add_url_rule(
    '{0}/'.format(context),  # URL 패스 정의
    view_func=serving,  # 서빙 클래스 객체
    methods=['GET', 'POST']  # 서비스 가능 HTTP 메소드
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


def configure_logging(app, format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'):
    if app.debug:
        return

    import logging
    from logging.handlers import RotatingFileHandler
    import os

    app.logger.setLevel(logging.INFO)
    info_log = os.path.join('./logs', 'info.log')
    info_file_handler = RotatingFileHandler(info_log, maxBytes=100000, backupCount=10)
    info_file_handler.setLevel(logging.DEBUG)
    info_file_handler.setFormatter(logging.Formatter(format))
    app.logger.addHandler(info_file_handler)


configure_error_handlers(app=app, code='99', msg_fn=message_format)
configure_logging(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

### 파일명은 app.py 로 변경하여 사용 ###
