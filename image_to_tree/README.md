image_to_tree
===================
한장의 이미지로 부터 그 안의 글자, 구조적 도형등을 식벽하여 구조를 트리로 만든다.



* ###square_tree
    이미지 전부를 담은 노드를 루트로 한다.
    
    node 의 구조 : Node(name=이름,parent=부모,position=[sx,sy,ex,ey,size,code],proportion=자신의 넓이/부모의 넓이,data=[여러 데이터])
    * name 
    
        node 의 이름
        * type : string
        * 포맷 : 
    * parent
    
        부모 node의 포인터
        * 포맷 : 노드
    * type
    
        현재 node의 타입
        
        타입 목록
        * element : node 속에 node 가 없는 최소한의 사각형
        * square :  node 속에 다른 node 가 존재
        * sentence : 글자를 포함한 node 일 경우
        * table : 여러개의 element 가 겹치지 않고 표를 이루는 특별한 경우
         
    * position
    
        이미지 상에서 노드가 존재하는 위치
        * sx : x 좌표의 좌측값 node 에서 가장 작은 x 좌표
        * sy : y 좌표의 좌측값 node 에서 가장 작은 y 좌표
        * ex : x 좌표의 우측값 node 에서 가장 큰 x 좌표
        * ey : y 좌표의 좌측값 node 에서 가장 큰 y 좌표
        * size : node 의 넓이
        * code :
        
    * proportion
    * data
    


