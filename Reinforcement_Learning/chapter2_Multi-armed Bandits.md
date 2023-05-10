# Chapter 2: Multi-armed Bandits

## 2.1 A k-armed Bandit Problem

강화학습이 다른 학습(ML, DL etc.)과 다른 점은 어떠한 행동을 했을 때, 그 행동에 따른 조치를 평가기재로 사용한다는 것이다.

이번 chapter에서는 전체 강화학습의 상태가 아니라 평가적 피드백만 다루는 간소한 nonassociative한 피드백을 다룬다.  
* nonassociative task - 행동과 상황이 어떠한 관련이 있을 필요가 없음
* associative task - 행동과 상황이 연관되어 최상의 행동이 상황에 따라 달라지는 경우

nonassociative task의 예시로 k-armed Bandit Problem를 사용한다. k-armed Bandit Problem이란 k 개의 서로 다른 옵션 또는 행동을 선택하고 각 행동이후에는 선택한 행동에 따라 정해진 확률 분포에 맞게 보상을 받는 문제이다.    
여기에서 알 수 있듯이, nonassociative task적인 문제를 예시로 들었듯이 정해진 확률 분포는 행동이 시행될때 마다 바뀌지 않는다. 만약, 행동이후 확률분포가 random하게 바뀌었다면 associative task를 사용하여 문제를 해결해야 한다.  

k-armed Bandit Problem에서 k개의 액션(action)은 선택될 때 기대되는 보상 또는 평균보상을 가지고 있고 이를 해당 액션이 가지고 있는 가치(the value of action)라고 한다.
이를 수식으로 나타내면 다음과 같다. t 시점에서의 액션은 $A_t$, 해당 액션을 통해 얻는 보상을 $R_t$ 한다.
  
![expected_reward](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/expected_reward.png?raw=true)
  

해당 수식은 각 액션의 값을 정확히 알고 있다면 가장 가치가 높은 액션을 선택하면 쉽게 풀리지만 문제는 우리가 정확한 액션의 가치를 알지 못하고 추정치만 알고 있다는 것이다. 시점 t에서의 액션 a의 추정가치를 $Q_t(a)$라고 하며 우리의 목표는 $Q_t(a)$가 $q_*(a)$에 가깝게 만드는 것이다.  
그렇다면 어떻게 하면 $Q_t(a)$가 $q_*(a)$에 가깝게 만들 수 있을까. 답은 exploration & exploitation에 있다.  

### exploration & exploitation

exploitation은 greedy한 방법이라고 생각하면 된다. 현 상태에서의 가장 최상의 보상을 위한 액션을 취하는 것이다. 하지만 이러한 방식은 당장은 보상이 적지만 추후에 커다란 보상을 획득할 수 있는 상황을 차단해 버린다. 이러한 상황을 방지하기 위해 exploration이 존재한다.  
exploration은 greedy한 방식보다는  천천히 다른 액션의 보상을 확인해 가면서 장기적인 관점에서의 보상 극대화를 기대하게 한다.   
서로 바라는 상황이 다르므로 exploration과 exploitation 사이에는 당연하게 trade off관계가 존재한다.

## 2.2 Action-value Methods

액션의 가치, 즉 $Q_t(a)$를 추정하는 Action-value Methods은 다음과 같은 수식으로 설명할 수 있다.  
![action_value](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/action_value.png?raw=true)
  
앞선 t시간동안 액션을 통해 얻은 보상/앞선 t시간동안 액션을 한 횟수이며 만약 분모가 0이면(아무런 시도가 없으면) $Q_t(a)$를 0과 같은 기본값으로 한다.  

만약, 위 식이 무한대로 커지면 대수의 법칙의 의해 그 값은 $q_*(a)$로 수렴한다. 이것을 **sample-averge**라고 한다.

여기서 액션을 선택하는 방법중 하나는 greedy한 방법을 채택하는 것이다.  
![greedy_action](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/greedy_action.png?raw=true)
  

이러한 액션은 앞서 말했듯이 현재의 보상을 극대화 하는 방법이지만, exploration이 없다는 단점이 있다.
이러한 단점을 보안하는 방법으로는 $\epsilon$-greedy 방법이 있다.


$\epsilon$-greedy이란 일정한 확률($\epsilon$)로 exploration을 하고 나머지 확률($1-\epsilon$)로써 exploitation을 하는 방법이다. 이러한 방법은 $\epsilon$에 의해 자연스럽게 exploration & exploitation를 하기 때문에 단순 greedy 액션보다는 효과적이다.  

## 2.3 The 10-armed Testbed

다음과 같은 k가 10일때의 bandit 문제를 확인해 보자. 
  
![reward_distribution](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/reward_distribution.png?raw=true)
  
이러한 분포를 가진 상황에서 $\epsilon$-greedy의 성능을 비교해 보자. $\epsilon$은 각각 0, 0.1, 0.01이며 결과는 다음과 같다.
  
![greedy_e_greedy](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/greedy_e_greedy.png?raw=true)
  
결과를 확인해 보면 greedy한 방법($\epsilon$ 이 0)보다 $\epsilon$-greedy의 성능이 우수하다는 것을 확인해 볼 수있다. 하지만, 반드시 $\epsilon$-greedy 방법이 greedy한 방법보다 더 좋다고는 할 수 없다. 앞선 분포에서 만약 분산이 0이라면 exploitation을 할 필요가 없기 때문에 greedy한 방법의 성능이 더욱 우수할 것이다.  

## 2.4 Incremental Implementation

앞서 설명한 Action-value Methods는 관측된 모든 보상의 샘플을 평균으로 Action-value를 추정했었다.
  
![action_value_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/action_value_2.png?raw=true)
  
수식을 보면 알 수 있듯이 $Q_n$을 계산하기 위해서는 n-1번째의 R값을 이용해야한다. 하지만 R의 값이 점차 많아지면 메모리의 한계가 올 수 있는 위험성이 있다. 이런 위험성을 방지하고자 다음과 같이 수식을 변형시킨다.
  
![action_value_3](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/action_value_3.png?raw=true)

이렇게 변형시킴으로써 $Q_n+1$값을 계산하기 위해서는 $Q_n$과 $R_n$ 값만 알면 된다. 이러한 방식을 Incremental Implementation라고 하며 다음과 같은 의미를 가진다.

$NewEstimate \larr OldEstimate + StepSize [Target − OldEstimate]$

여기서 [Target − OldEstimate]은 error라고 하며 Target이 목표쪽으로 움직이며 작아질수 있다. 즉, $Q_n$과 $R_n$의 오차를 의미한다.  
StepSize는 n의 갯수(표본)의 반비래 하며 시간 단계마다 달라질 수 있다. 즉, StepSize를 조절하며 $Q_n$을 수렴시킬 수 있다.  
다음은 Incremental Implementation의 의사코드이다.
  
![a_simple_bandit_algorithm](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/a_simple_bandit_algorithm.png?raw=true)
  
## 2.5 Tracking a Nonstationary Problem

앞서 설명한 bandit problem은 시간에 따라 보상확률이 변하지 않는 문제(Stationary Problem)였다. 이번 장에서는 시간에 따라 보상확률이 변하는 Nonstationary Problem을 다룬다. Nonstationary Problem에서의 합리적인 해결방식은 오래된 보상보다 최근 보상에 더 많은 가중치를 둔다는 것이다.(단순하게 생각해 보면 액션을 취할때 마다 보상활률이 변하면 당연히 최근에 얻는것이 제일 중요할 것이다.)
  
![action_value_3](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/action_value_3.png?raw=true)
  
다음과 같은 Action-value를 다음과 같이 변형시켜보자.

![action_value_4](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/action_value_4.png?raw=true)
  
식을 확인해 보면 stepsize인 $\alpha$(단, $\alpha$ 는 (0,1])에 따라 $Q_1$와 $R_i$의 가중이 변하는 것을 확인 할 수 있다.  
이것을 **weighted-average(가중 평균)** 이라고 하며 (1-α)ⁿ + ∑ⁿⱼ₌₁ α(1-α)ⁿ⁻ʲ = 1임을 확인해 볼 수 있다.

가중 평균이 의미하는 바는 다음과 같다. $R_i$에 할당된 가중치인 α(1-α)ⁿ⁻ⁱ는 $\alpha$가 1보다 작기 때문에 보상의 개수가 증가함에 따라 감소한다. 따라서 Nonstationary Problem을 다루는데 필요한 오래된 보상의 가중치의 감소를 확인할 수 있다.

$\alpha$ 값에 대해서 매 step마다 다르게 설정하는것이 좋을 수도 있다. stochastic approximation theory에서 확률 1로 수렴하게 하는 $\alpha$의 조건을 다음과 같이 정의했다.

![alpha_size](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/alpha_size.jpg?raw=true)
  

첫번째 조건은 step이 초기조건이나 random한 변함에도 잘 견더낼 정도로 커야한다는 의미이고 두번째 조건은 수렴하기 위해서는 충분히 작아야 한다는 것이다.

## 2.6 Optimistic Initial Values

앞선 Action-value Methods의 식을 확인해 보면 초기 Action-value 값인 $Q_1$에 따라 sample-average, weight average등의 값이 달라지는것을 알 수 있다. 이를 biased(편향되었다)라고 말한다. 이런 초기값을 지정해 줌에 따라 exploration를 하도록 하는 효과를 얻을 수 있다.

앞서 예시로 들었던 10-armed testbed를 생각해 보자. $Q_1$의 값을 +5로 준다고 하면 보상은 5보다 작으므로 exploration를 유도할 수 있다. greedy 방법이라 할지라도 무조건 한번은 exploration를 할것이다.다음은 $Q_1$의 값을 +5로 준상태에서 greedy와 $\epsilon$-greedy의 성능 비교이다.

![initial_values](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter2/initial_values.png?raw=true)
  
하지만 이러한 Initial Values의 선택은 stationary한 환경에서만 좋을뿐이다. 계속 변하는 nonstationary한 환경에서는 이전의 값은 의미가 없어지기 때문이다.
