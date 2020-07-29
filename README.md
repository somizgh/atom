시스템 환경
cuda 10.2
python 3.6
tensor-gpu 2.0



## 일반 파일 제작 규칙
1. 파일의 이름은 만들고자 하는 기능을 적는 것이 바람직하다.
예시) image_to_tree : 이미지를 트리로, Dictionary : 단어 사전
2. 일반 파일속에는 모델파일을 제작하지 않고 최상단의 atom 디렉토리에 제작 한다.



## 모델 파일 제작 규칙
1. 모델 제작, 검증파일의 경우 atom 디렉토리에 역활뒤에 model을 붙인다.
예시) predict_next_character_model
    다음에 나올 자모나, 음을 예측하는 모델
   
2. 폴더의 이름을 A 라고하면 폴더안에 A 에서만 사용되는 변수파일의 이름은 A_variables 이다.
3. 모델 제작파일의 이름은 A_maker 이다. 예시) predict_next_character_model_maker
4. 모델을 로드하고 사용하는 함수들이 있는 파일은 폴더의 이름에서 model을 뺀다.
    예시) predict_next_character

5. 위의 파일을 import 시 대문자 약자로 import 하고 variables 파일은 V, maker 파일은 M을 붙인다.
    예시) import predict_next_character_model_variables as PNCMV
    
 
    
