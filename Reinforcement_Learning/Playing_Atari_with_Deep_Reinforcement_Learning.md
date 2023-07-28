
## DQN은 sensory input으로 부터 정책을 곧바로 배우는 첫번째 심층강화학습 모델. 
  
## DQN = alexnet을 사용한 cnn + q-learning

rl + cnn은 원래 있었지만 DQN은 이것을 더 발전시킴.

## DQN이 해결한 3가지 문제

1. sensory input을 rl 에이전트에 적용방법
2. sample 간에 시간적 연관성을 줄이는 방법
3. moving q-target

## 해결한 방법  

1. cnn을 이용해여 차원의 저주 극복

    DQN에서는 이미지 전처리 부분에서 과거의 상태를 추가함.

    * 과거의 상태를 사용한 이유  
    1. RL 알고리즘에서 성능을 높이기 위해
    2. non-MDP 를 MDP로 바꾸기 위해
  
  
2. experience replay을 통한 샘플관의 시간연관성 극복  
  
    문제점
    * q-learing은 Q(s,a)를 학습할 때 불안정 할 수 밖에 없음.
      -> 학습 초기에는 Q(s,a)와 다음 상태 추산값인 Q(s',a')의 추산이 부정확하기 때문에
    * q-learing의 argmax Q(s',a') 값이 매우 높게 추산됨
    * $\epsilon$ - greedy 정책은 한번의 업데이트로 매우 크게 바뀔 수 있다.
    * MDP 자체가 순차적 진행됨

    해결법  
    * (mini)batch 로 Q(s,a) 업데이트하기
    * experience replay를 이용하여 (mini)batch update를 수행
    * experience replay : replay buffer( $D$ )라는 FIFO que에 과거의 샘플을 넣고 업데이트 시에 꺼내어 사용하는 방법
    * experience replay 주의점 
        1. off-policy 기법에만 사용가능
        2. batch size에 관해서는 아직 정해진게 없음
  
3. target network를 도입하여 moving target issue를 줄임
    함수근사가 없는 q-learning의 경우에는 하나의 업데이트가 다른 Q(s,a)의 영향을 미치지 않음.
    함수근사가 있을 경우엔 영향을 미침 -> target이 점점 이동해서 도달할 수 없는 문제가 생김.
    이러한 문제를 해결하기 위해 DQN은 target network를 도입.

    target network $\hat{Q_\theta}$ : $Q_\theta$ network의 복사판

    c번 업데이트 동안, $\hat{Q_\theta}$ 를 이용해서 Q 타겟을 계산하고 그 값을 활용해 $Q_\theta$ network 값을 업데이트
    c번 업데이트 후, $Q_\theta$ network 값을 $\hat{Q_\theta}$ 에 적용  


