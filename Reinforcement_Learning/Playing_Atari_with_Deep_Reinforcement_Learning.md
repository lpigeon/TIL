

dqn은 sensory input으로 부터 정책을 곧바로 배우는 첫번째 심층강화학습 모델

dqn = alexnet을 사용한 cnn + rl

dqn이 해결한 3가지 문제

1. sensory input을 rl 에이전트에 적용방법
2. sample 간에 시간적 연관서을 줄이는 방법
3. moving q-target

dqn 한눈에 보기

차원의 저주 완화 <- deep cnn 활용
rl 에이전트의 학습과정 안정화
1. experience replay
2. target network -> nature 버전논문에 자세히 기술

rl + cnn은 원래 있었지만 dqn은 이것을 더 발전시킴

1. cnn을 이용해여 차원의 저주 극복
2. experience replay 시간연관성 극복
3. target network를 도입하여 moving target issue를 줄임

1. 
2. q-learing은 Q(s,a)를 학습할 때 불안정 할수 밖에 없음 