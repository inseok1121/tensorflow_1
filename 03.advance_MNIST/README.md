
## CNN(Convolution Neural Network)

### 과정

  1. Feature Extraction
  2. Shift and distortion Invariance
  3. Classification

  이 과정을 여러번 반복적으로 수행하여 Local feature로 부터 Global feature을 얻어낸다.

  1번과 2번을 위해  filter와 sub-sampling을 거침

    ```
    filter
      - Convolution

    sub-sampling
      - max-pooling : 영역에서 가장 큰 값 선택
      - average-pooling : 영역 평균 값
    ```
