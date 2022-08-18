# Naive_model
The naive model for medication recommendation


[Process]

1. 시점마다 진단/수술 정수 인코딩된 벡터를 진단/수술 덴스 벡터로 변환
- 옵션: 정수 인코딩된 벡터, 멀티-핫 벡터

2. 시점마다 있는 진단/수술 덴스 벡터들을 평균으로 하여 하나의 진단/수술 덴스 벡터로 변환
- 옵션: sum, concat, mean

3. 진단/수술 덴스 벡터를 concat하여 하나의 덴스 벡터로 변환
- 옵션: sum, concat, mean

4. 덴스 벡터를 2개의 FC layer를 통해 최종 의약품 차원의 벡터로 변환
