


## MNIST

MNIST는 간단한 컴퓨터 비전 데이터
손으로 쓴 이미지의 정보 파악

MNIST 데이터는 학습 데이터(mnist.train, 55000), 테스트 데이터(mnist.test, 10000), 검증 데이터(mnist.validation, 5000)로 이뤄져있다.

각 데이터 셋은 두 부분으로 나뉜다.
 - 손으로 쓴 숫자 이미지 : xs  (mnist.train.images)
 - 그에 따른 라벨 : ys (mnist.train.labels)


 `
 one-hot vector : 목적인 Value에 1을, 다른 것들엔 0을 부여하는 vector 표현 방식
 `

#### Softmax Regression

서로 다른 여러 항목 중 하나일 확률을 계산할 때 활용

Softmax Regression은 두 단계로 이뤄져있다.

  1. 데이터가 각 클래스에 속한다는 Evidence를 수치적으로 계산
  ![evidence](https://github.com/inseok1121/tensorflow_1/blob/master/images/softmax_evidence.png)

  2. 계산한 값을 확률로 변
