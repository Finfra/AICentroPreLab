"""
SACP AI 포탈과 연계를 위한 기본 객체 생성
"""
from aicentro.session import Session
sacp_session = Session(verify=False)

# 배치 전용 설정 정보 로드
from aicentro.config import BatchConfig
batch_config = BatchConfig()
# 배치서비스 개발 시 INPUT 디렉토리 절대 경로 구하기
input_dir = batch_config.input_dir
output_dir = batch_config.output_dir


"""
배치 코드 작성
"""
...


"""
[Optional] 결과 생성 전 필요 시 기존 결과 파일 삭제 
"""
import os
# 파일 유무 확인 후 삭제하기.
if os.path.isfile(output_dir + '/output.csv'):
    os.remove(output_dir + '/output.csv')

"""
[Optional] 결과 생성 후 필요 시 현재 입력 파일 삭제 (또는 압축 후 이름 변경)
"""
import os
import zipfile
from datetime import datetime

# Backup Folder 만들기
if not os.path.isdir(os.path.join(input_dir, 'backup')):
    os.mkdir(os.path.join(input_dir, 'backup'))

input_zip = zipfile.ZipFile(os.path.join(input_dir, 'backup', datetime.today().strftime('%Y%m%d') + '.zip'), 'w')
for folder, subfolders, files in os.walk(os.path.join(input_dir)):
    for file in files:
        if file.endswith('.csv'): # 확장자 입력
            input_zip.write(os.path.join(folder, file),
                            os.path.relpath(os.path.join(folder, file), input_dir),
                            compress_type=zipfile.ZIP_DEFLATED)

