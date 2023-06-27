# Chapter 6: Temporal-Difference Learning  

Temporal-Difference(TD) = Monte Carlo + Dynamic Programming  

TD는 MC와 DP 아이디어의 결헙이라고 하는 만큼 MC의 명확환 환경모델이 없어도 학습할 수 있다는 점과 DP처럼 최종 결과를 기다리지 않고 다른 학습된 추정치를 기반으로 추정치를 업데이트 하는 방식(Bootstrap)을 모두 가지고 있다.  

## 6.1 TD Prediction  

우선, MC와 TD의 차이점을 알아보자.  

MC의 $V(S_t)$ 업데이트 방식은 다음과 같다.  

![mc_v_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/mc_v_update.png?raw=true)  
  
위 식에서 $G_t$는 시간 t 이후의 실제 반환값이며 이 방법을 Constant-Step MC 방법이라고 한다. 이러한 방법은 $G_t$를 계산하기 위해 에피소드가 끝날때 까지 기다려야 한다(그때에만 Gt가 알려진다).  

이러한 단점을 보안하기 위해 TD는 바로 다음 스템까지만 기다리는 업데이트 방식을 취한다. 다음은 TD의 $V(S_t)$ 업데이트 방식이다.  

![td_v_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/td_v_update.png?raw=true)  

위 식을 살펴보면 $G_t$ 대신 $R_{t+1}$ + $\gamma$ $V(S_{t+1})$이 있는 것을 확인할수 있다. 이렇게 함으로써 TD는 t + 1 시간에 즉시 목표를 관측하고 보상을 사용하여 업데이트가 가능하다. 위 방법을 TD(0) 또는 $one-step$ TD 방법이라고 하며 특수한 형태로 TD($\lambda$), $n-step$ TD 방법이 있다(추후에 다룸).

위 두 식에서의 차이점에서 확인 할 수 있듯이 MC와 TD의 업데이트 목표는 각각 $G_t$ 와 $R_{t+1}$ + $\gamma$ $V(S_{t+1})$이다. 아래는 TD(0)의 절차적 형태이다.  

![tabular_td(0)](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/tabular_td(0).png?raw=true)  

앞선 설명에서 TD는 MC와 DP가 적절하게 섞인 방식이라고 했다. 아래 식에서 MC는 (6.3)의 추정치를 목표(기대 반환 대신 샘플 반환)로 사용하고 DP는 (6.4)의 추정치를 목표(bootstrap)로 사용한다. 두 방법 모두 기대값이 아닌 추정치를 목표로 사용한다. 이것은 TD에서도 나타나는데 TD는 DP와 마찬가지로 (6.4)의 기대값을 샘플링하며 실제 $v_\pi$ 대신 현재 추정치인 V를 사용한다. 따라서, TD는 MC의 sampling과 DP의 bootstrapping의 결합이다.  

위 식에서 TD(0)의 업데이트 괄호안의 $R_{t+1}$ + $\gamma$ $V(S_{t+1})$ - $V(S_{t})$는 두 추정치의 차이를 측정하는 오차이며 이를 TD $error$라고 부르며 다음과 같이 나타낸다.  

![td_error](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/td_error.png?raw=true)  

이러한 TD $error$는 다음 상태와 다음 보상에 의존하기 때문에 실제로는 t시간의 TD $error$는 t+1시간에서 사용가능하다. 또한, 에피소드 동안 $V$의 배열이 바뀌지 않는다면 MC $error$는 TD $error$의 합으로 표현할 수 있다. 아래 식은 step 크기가 작으면 근사적으로 성립할 수 있다.  

![mc_error](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/mc_error.png?raw=true)  

## 6.2 Advantages of TD Prediction Methods  

TD 방법의 이점은 위에서 MC와 DP와 비교하면서 설명했다. DP의 환경을 완벽히 알아야 하는 단점을 없애고 MC의 에피소드가 완전히 끝나야한 한다는 단점을 모두 해결한 방법이 TD이다. 하지만 TD는 확실하게 수렴할까? 이러한 질문의 답은 그렇다 이다. 앞서 보았던 TD(0)는 수렴한다는 것이 수학적으로 증명되었다.  

TD(0)의 수렴 특성은 step 크기 매개변수에 따라 달라지는데 이 step크기가 상수이고 충분히 작다면, TD(0)는 평균적으로 $v_\pi$ 에 수렴한다. 즉, 평균적으로 추정된 값들은 실제 값에 수렴한다. 또한, step 크기가 일반적인 확률 근사 조건에 따라 감소한다면 TD(0)는 확률 1로(반드시) $v_\pi$로 수렴한다.  

그렇다면 만약 TD와 MC 방법 모두 수렴한다면 무엇이 먼저 수렴에 도달하는지에 관한 질문이 있을 수 있다. 이 질문에 대한 답은 아직 수학적으로 증명한 사례는 없지만 일반적인 관찰로써 TD가 constant-$\alpha$ MC 방법보다 더 빠르게 수렴하는것으로 알려져있다.  

## 6.3 Optimality of TD(0)  

만약 경험이 제한된 상황에서 점진적인 학습방법의 일반적인 접근 방법은 경험을 반복적으로 제공하여 방법이 수렴할때 까지 답을 찾는 것이다. 근사값 함수 $V$ 가 주어지면, 각 시간 step t에서 비종단 상태를 발견할 때마다 앞선 업데이트 식에 의해 증분을 계산한다. 하지만 가치 함수는 이러한 증분의 합으로만 한 번만 변경된다. 그런 다음 사용 가능한 모든 경험은 새로운 값 함수로 다시 처리되어 전체적인 증분을 생성하는 데 사용된다. 이 과정을 값 함수가 수렴할 때까지 반복하는데 이러한 방식은 배치 업데이트(batch updating)라고도 불리며, 업데이트는 모든 훈련 데이터를 한 번 처리한 후에만 수행된다.  

이러한 배치 업데이트에서 TD(0)는 step 크기가 충분히 작게 선택되었을 경우, 결정론적으로 단일한 담에 수렴한다. constant-$\alpha$ MC 방법 또한 동일한 조건에서 경정론적으로 수렴하지만 두 방법은 같은 답에 도달하지 않는다. 이러한 결과를 이해하기 위해서 몇가지 예를 들어 설명한다.  

Example 6.2 Random Walk의 문제를 보면 다음과 같은 환경에서 action이 없는 MDP인 MRP(Markov reward process)를 사용하여 각 가치함수를 구하고 MC와 TD를 비교하였다.  

![random_walk](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/random_walk.png?raw=true)  

![random_walk_td_mc](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/random_walk_td_mc.png?raw=true)  

왼쪽 그래프는 TD(0)의 단일 실행에서 다양한 에피소드 후 학습된 값들을 보여주고 오른쪽 그래프는 MC와 TD 방법의 학습된 값 함수와 실제 값 함수 사이의 제곱근 평균 제곱 오차(RMS 오차)를 보여준다. 이 작업에서 TD 방법이 일관되게 MC 방법보다 더 나은 성능을 보였다.  

위와 같은 완벽하게 동일한 환경에서 batch TD(0)와 batch MC 방법의 수렴 결과값이 달라지는 이유는 batch MC는 항상 학습 데이터셋에서 평균 제곱 오차를 최소화하는 추정값을 찾는 반면, 배치 TD(0)는 항상 Marcov process의 최대 우도 모델에 대해 정확히 올바른 추정값을 찾기 때문이다. 일반적으로 최대 우도 추정은 데이터를 생성할 확률이 가장 큰 매개변수 값을 의미한다. 이 경우, 최대 우도 추정은 관측된 에피소드에서 자명한 방식으로 형성된 Markov 과정 모델이다.  

이 모델을 가지고 모델이 정확하다고 가정할 경우 정확하게 올바른 값 함수의 추정값을 계산할 수 있으며 이를 확실성-등가 추정값(certainty-equivalence estimate)이라고 한다. 이는 기저 과정의 추정값이 근사되는 대신에 확실하게 알려져 있는 것으로 가정하는 것과 동일하다. 일반적으로 batch TD(0)는 확실성-등가 추정값으로 수렴한다.  

이러한 이유 때문에 TD방법이 MC 방법보다 빠르게 수렴하는 것이다. 하지만, 확실성-등가 추정값이 어떤 의미에서 최적의 해결책이지만, 직접적으로 계산하는 것은 거의 불가능하다. 만약, n이 상태의 수를 나타낸다고 할때 프로세스의 최대 우도 추정치를 형성하는 것만으로도 n^2의 메로리가 필요하며 계산하는 것은 n^3의 계산 단계가 필요하다. 이러한 불가능 성을 TD방법을 사용하여 근사화 할 수 있다는 것이 중요한 사실이다.  

## 6.4 Sarsa: On-policy TD Control  

TD 방법을 제어 문제에서 사용하기 위해서 GPI 정책 패턴을 따를 필요가 있다. 평가 또는 예측 부분에 TD 방법이 사용되는데 MC 방법과 마찬가지로 탐사와 활용의 균형을 맞추어야 한다. 이 장에서는 on-policy TD 제어 방법을 다룬다.  

제어의 첫번째 단계는 상태 가치 함수가 아닌 행동 가치 함수를 학습하는 것이다. on-policy 방법에서는 현재의 행동 정책 $\pi$에 대해 모든 상태 s 와 행동 a의 대한 $q_\pi(s,a)$를 추정해야 한다. 이러한 방식은 아래와 같이 번갈아 가며 수행된다.  

![state_action_pairs](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/state_action_pairs.png?raw=true)  

이러한 방식을 식으로 나타내면 다음과 같다.  

![td(0)_value](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/td(0)_value.png?raw=true)  

이러한 업데이트는 빕종료 상태 $S_t$에서의 각 전이 후에 수행되며 $S_{t+1}$이 종료 상태이면 $Q(S_{t+1},A_{t+1})은 0으로 정의된다. 이 규칙은 상태-행동 쌍에서 다음으로 전이하기 위해 사용되는 (St,At,Rt+1,St+1,At+1) 다섯 가지 이벤트의 모든 요소를 사용하는데 이 다섯 가지 이벤트의 집합 때문에 이 알고리즘은 $Sarsa$라고 부른다.  

$Sarsa$ 알고리즘은 지속적으로 $\pi$에 대한 $q_\pi$를 추정하고 동시에 $q_\pi$에 대한 탐욕성을 가진 $\pi$로 변경된다. $Sarsa$ 알고리즘의 수렴 특성은 정책이 Q에 대해 어떻게 의존하는지에 따라 다른데 예를 들어, $\epsilon$-greedy 정책이나 $\epsilon$-soft 정책을 사용할 수 있다. 상태 행동 쌍이 무한히 반복되고 정책이 탐욕 정책으로 한계에서 수렴하는 경우 $Sarsa$는 확률 1로 최적 정책과 행동 가치 함수에 수렴한다.  

![sarsa](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/sarsa.png?raw=true)  

## 6.5 Q-learning: Off-policy TD Control  

강화학습에서 초기의 중요한 발전 중 하나는 Watkins (1989)에 의해 정의된 off-policy TD 제어 알고리즘인 Q-learning의 개발이다.  

![q_learning](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/q_learning.png?raw=true)  

위의 식은 학습된 행동 가치 함수인 $Q$를 따르는 정책과는 독립적으로 최적의 행동 가치 함수 $q_*$를 직접 근사화 한다. 이는 알고리즘의 분석을 현저하게 단순화시키고 초기의 수렴 증명을 가능하게 했다. 정책은 여전히 영향을 미치며, 어떤 상태-액션 쌍이 방문되고 업데이트되는지를 결정하지만 올바른 수렴을 위해서는 모든 쌍이 계속해서 업데이트되는 것만 필요하다. 아래는 Q-learning의 알고리즘이다.  

![q-learning_algorithm](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/q-learning_algorithm.png?raw=true)  

## 6.6 Expected Sarsa  

Q-learning과 유사하지만 다음 상태 행동 쌍에 대한 최댓값을 사용하는 대신 현재 정책에서 각 행동의 가능성을 고려하여 기대값을 사용하는 학습 알고리즘을 고려해보자. 해당 알고리즘의 식은 다음과 같을 것이다.  

![expected_sarsa](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/expected_sarsa.png?raw=true)  

위 알고리즘은 Q-learning의 구조를 따르지만 기대값에 따라 결정론적으로 움직이므로 Expected $Sarsa$ 라고 불린다. 아래는 Q-learning과 Expected $Sarsa$의 비교 다이어그램이다.  

![q_expected_sarsa_diagram](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/q_expected_sarsa_diagram.png?raw=true)  

Expected $Sarsa$는 $Sarsa$보다 계산적으로 더 복잡하지만, 그 대신 $A_{t+1}$의 무작위 선택으로 인한 분산을 제거한다. 같은 양의 경험을 가지고 있다면 Expected $Sarsa$가 $Sarsa$보다 약간 더 좋은 성능을 발휘할 것으로 기대할 수 있으며, 실제로 일반적으로 그러하다. 다음은 Expected $Sarsa$, $Sarsa$, Q-learning을 비교한 cliff-walking 과제의 결과를 보여준다.  

![cliff_walking](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter6/cliff_walking.png?raw=true)  

이 cliff-walking 결과에서 Expected $Sarsa$는 on-policy으로 사용되었지만, 일반적으로 목표 정책 $\pi$와 다른 정책을 사용하여 행동을 생성할 수 있으며, 그 경우 off-policy 알고리즘으로 변환된다. 예를 들어, $\pi$가 탐색적인 행동을 생성하는 greedy 정책인 경우, Expected $Sarsa$ 정확히 Q-learning이 된다. 이런 의미에서 Expected $Sarsa$는 Q-learning을 포함하고 일반화시키면서도 Sarsa보다 신뢰성있게 개선됩니다. 작은 추가적인 계산 비용을 제외하고는 Expected $Sarsa$가 더 잘 알려진 TD 제어 알고리즘인 $Sarsa$와 Q-learning을 완전히 지배할 수 있다.