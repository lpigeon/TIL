# Chapter 4: Dynamic Programming  

Dynamic Programming(DP)는 환경을 MDP로 완벽한 모델링한 경우에 최적 정책을 계산할 수 있는데 사용할 수 있는 알고리즘이다. 전통적인 DP는 완벽한 모델과정과 컴퓨팅 비용문제로 인해 제한적으로 유용하다. 이번 장부터 환경은 일반적으로 유한한 MDP라고 가정한다.  
강화학습과 마찬가지로 DP의 주요 아이디어는 좋은 저액을 찾기 위해 가치 함수를 사용하여 탐갯을 조직화 하고 구조화 하는것이다.  

## 4.1 Policy Evaluation (Prediction)

DP에서 임의의 정책 $\pi$에 대한 상태 가치 함수 $v_\pi$를 계산하는 것을 $policy$ $evaluation$(정책 평가) 또는 $predicton$ $problem$(예측 문제)이라고 한다.  

![policy_evaluation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/policy_evaluation.png?raw=true)  

만약 환경의 동적인 특성이 완전히 알려져 있다면, 위 식은 |S|개의 동시 선형 방정식으로 이루어진 |S|개의 방정식일 것이다. 이러한 방정식을 푸는 것은 지루한 일이고 우리가 원하는 것은 반복적인 해결 방식이기 때문에 다음을 생각해 보자. $v_0$, $v_1$, $v_2$ ...라는 근사적인 가치 함수가 있다고 한다면 초기 근사치인 $v_0$은 임의로 선택될 수 있다. 단, 종단 상태가 있다면 그 값은 0이어야 한다. 그렇다면 각 차례의 근사치는 다음과 같은 벨만 방정식 규칙을 따른다.  
  
![Bellman_equation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/Bellman_equation.png?raw=true)  
  
위 업데이트 규칙은 벨만 방정식에 의해 $v_k$ = $v_\pi$가 fixed point가 된다. 또한, $v_\pi$의 존재를 보장하는 조건하에 k가 무한대로 갈대 $v_k$는 $v_\pi$로 수렴한다. 이러한 알고리즘을 $iterative$ $policy$ $evaluation$(반복적인 정책 평가)라고 한다.  
위 식을 살펴보면, $v_k$로 부터 $v_{k+1}$을 얻기 위해 반복적인 계산을 한다는 것을 알 수 있다. 이러한 종류의 연산을 $expected$ $update$라고 한다.  

위 식을 사용하여 반복적인 컴퓨팅계산을 하게 된다면 두가지 방식을 생각해 볼 수있다.  
첫번째는 두개의 배열을 사용하여 하나에는 $v_k(s)$의 값을 다른 하나에는 $v_{k+1}(s)$의 값을 넣는 방식이다.  
두번째는 하나의 배열을 사용하여 $v_k(s)$를 통해 계산된 $v_{k+1}(s)$을 "원래 자리"에 업데이트 하는 방식이다.  
이 두 방식중 원래 자리로 업데이트 하는 두번째 방식이 더 빠르게 수렴하는 경우가 많다. 이러한 현상은 새로운 데이터를 사용 가능한 즉시 사용할 수 있기 때문이다.(업데이트는 상태 공간을 통해 스윕(sweep)하는 과정에서 수행된다고 생각한다.)  
따라서, DP 알고리즘을 사용할 때, "원래 자리" 버전을 고려한다. 다음은 반복적인 정책 평가의 완전한 "원래 자리" 버전 의사 코드이다.  

![Iterative_Policy_Evaluation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/Iterative_Policy_Evaluation.png?raw=true)  

## 4.2 Policy Improvement  

정책의 가치 함수를 계산하는 이유는 더 나은 정책을 찾기 위해서이다. 그렇다면 어떤 정책을 따르는게 더 좋다는 것을 알 수 있을까. 이러한 질문의 대한 한가지 방법은 s에서 a를 선택하고 그 이후에 기존 정책을 따라가는 것을 고려한는 것이다. 즉, 행동 가치 함수를 통해 확인하는 방식이다. 다음은 행동 가치 함수의 식이다.  

![action_value_function](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/action_value_function.png?raw=true)  

만약, $q_\pi(s,a)$의 값이 $v_\pi$ 보다 크다면 새로운 정책을 따라가는 것이 기존의 정책을 따라가는 것보다 더 좋다고 보는 것이다. 이런 내용을 $policy$ $improvement$ $theorem$(정책 개선 정리)라고 한다.  

더 자세하게 말하면 정책 $\pi$와 $\pi^"$를 생각해 보자. 만약 아래와 같은 식이 성립하면 $\pi^"$가 $\pi$보다 더 나은 정책이라는 것이다.

![policy_improvement_theorem](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/policy_improvement_theorem.png?raw=true)  

![policy_improvement_theorem_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/policy_improvement_theorem_2.png?raw=true)  

위 식에 대한 증명은 다음과 같다.  

![the_proof_of_the_policy_improvement_theorem](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/the_proof_of_the_policy_improvement_theorem.png?raw=true)  

이러한 방식을 사용하여($q_\pi(s,a)$를 사용한) 새로운 탐욕 정책 $\pi$를 고려할 수 있다.  

![new_greedy_policy](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/new_greedy_policy.png?raw=true)  

위 방식처럼 원래 정책의 가치 함수에 따라 탐욕적으로 만들어진 새로운 정책을 만드는 과정을 $policy$ $improvement$(정책 개선)이라고 한다.  

만약 새로운 정책이 기존의 정책과 같다면 정책을 따르는 가치함수는 같을 것이고 그 식은 벨만 방정식과 동일하다.

## 4.3 Policy Iteration  

앞선 장에서 설명한 방식으로 정책을 결정한다면 다음과 같이 나타낼 수 있다.

![policy_iteration](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/policy_iteration.png?raw=true)  

이러한 방식을 $policy$ $iteration$(정책 반복)이라고 한다. 유한한 MDP는 유한한 수의 결정론적 정책을 가지기 때문에 위 알고리즘은 유한한 반복 횟수 내에 최적 정책과 최적 값 함수에 수렴해야 한다. 다음은 $policy$ $iteration$의 의사결정 코드이다.  

![policy_iteration_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/policy_iteration_2.png?raw=true)  

## 4.4 Value Iteration  

정책 반복의 단점 중 하나는 각 반복이 정책 평가를 포함한다는 것이다. 정책 평가를 반복적으로 수행하는 경우, $v_\pi$로의 정확한 수렴은 한계(limit)에서만 일어난다. 그렇다면 정확한 수렴을 기다리는 것이 맞을까 아니면 그 전에 중단하는것이 맞을까. 다음은 정확한 수렴전에 중단해도 되는 가능성을 보여준다.  

![Convergence_of_iterative_policy_evaluation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/Convergence_of_iterative_policy_evaluation.png?raw=true)  

정책 반복의 정책 평가 단계는 중간 중단 말고도 다른 방법으로 줄일 수 있다. 정책 평가가 한번의 훑기 후(각 상태의 업데이트 한 번)에 중단되는 특별한 경우 알고리즘인 $Value$ $Iteration$(가치 반복)이다. 다음은 가치 반복의 계산식을 보여준다. 앞서 설명한 정책 평가(evaluation)와는 다르게 모든 가치 함수중에 max값만을 취하여 greedy하게 업데이트한다는 것이 가치 반복의 아이디어이다.  

![value_iteration](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/value_iteration.png?raw=true)  

아래는 가치 반복의 의사결정코드이다.  

![value_iteration_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/value_iteration_2.png?raw=true)  

## 4.5 Asynchronous Dynamic Programming  

지금까지의 DP 방법의 단점은 MDP의 전체 상태 집합에 대한 작업을 수반한다는 것이다. 상태 집합이 커지면 커질수록 전체 스윕하는데는 엄청난 비용과 시간이 든다.  
Asynchronous DP(비동기 DP) 알고리즘이란 시스템적인 상태 집합의 스윕에 기반하지 않고, 그 어떤 순서로든 상태의 값을 업데이트하는 in-place iterative DP 알고리즘이다. 비동기 DP는 다른 상태의 값이 어떤 값이든 이용 가능한 경우에 상관없이 상태의 값을 업데이트한다. 어떤 상태의 값은 다른 상태의 값이 한 번 업데이트되기 전에 여러 번 업데이트될 수 있다. 이러한 것은 비동기 DP 알고리즘이 업데이트 할 상태를 선택하는데 큰 유연성을 제공한다.  
또한, 비동기 DP는 계산과 실시간 상호작용을 조합하는 것도 용이하게 만든다. 주어진 MDP를 해결하기 위해 반복적인 DP 알고리즘을 실행하는 동시에 에이전트가 실제로 MDP를 경험할 수 있다. 에이전트의 경험은 DP 알고리즘이 업데이트를 적용하는 상태를 결정하는 데 사용될 수 있다.  

## 4.6 Generalized Policy Iteration  

앞서서 설명 했듯이 정책 반복은 두가지 동시적으로 상호작용하는 프로세스로 구성된다.

- 정책 평가(policy evaluation) : 현재 정책에 따라 값을 일관되게 만드는 것
- 정책 개선(policy improvement) : 현재 값 함수에 대해 탐욕적인 정책을 만드는 것  

정책 반복에서는 위 두가지 프로세스가 번갈아가며 진행되지만, 사실은 필수적이지 않다. 가치 반복(value iteration)에서는 정책 개선 사이에 정책 평가의 단일 반복만 수행된다. 또한, 비동기 DP 방법에서는 평가와 개선 프로세스가 더 세분화된 단위로 교차되어 진행된다. 하지만 마찬가지로 두 프로세스가 모든 상태를 계속해서 업데이트하기만 한다면 최종 결과, 즉, 최적 가치 함수와 최적 정책으로 수렴한다.  

Generalized Policy Iteration(일반화된 정책 반복, GPI)란 정책 평가와 정책 개선 프로세스가 서로 상호작용하도록 하는 개념을 말한다. 이 개념은 두 프로세스의 세분화 및 기타 세부 사항과는 독립적으로 적용될 수 있다. 대부분의 강화학습 알고리즘은 GPI로 잘 설명될 수 있다. 즉, 모든 알고리즘은 식별 가능한 정책과 값 함수를 가지며, 정책은 항상 값 함수에 대해 개선되고 값 함수는 항상 해당 정책의 값 함수로 수렴하도록 작동한다. 다음은 GPI를 시각화한 그림이다.  

![GPI](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/GPI.png?raw=true)  

두 프로세스가 안정화되면, 가치 함수와 정책은 최적이어야 한다. 가치 함수는 현재 정책과 일관되어야 할 때 안정화되며, 정책은 현재 값 함수에 대해 탐욕적이어야 할 때 안정화된다. 따라서 두 프로세스는 정책이 자체 평가 함수에 대해 탐욕적인 정책을 찾았을 때에만 안정화되고 이는 벨반 최적 방정식이 성립하고, 따라서 정책과 가치함수가 최적임을 의미한다.  

GPI를 또 다른 관점에서 이해할 수 있는데, 바로 경쟁과 협력이다. 평가와 개선 프로세스는 서로 상반된 방향으로 작용하기 때문이다. 정책을 가치 함수에 대해 탐욕적으로 만들면 일반적으로 가치 함수는 변경된 정책에 대해 부정확해지고, 가치 함수를 정책과 일관되게 만들면 해당 정책이 더 이상 탐욕적이지 않아질 수 있다. 따라서, 두 프로세스는 상호작용하여 단일한 해결책을 찾아야한다.  

다음은 GPI에서 평가, 개선 프로세스를 2차원 공간의 두 선으로 표현한 것이다.  

![GPI_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter4/GPI_2.png?raw=true)  

위 기하학구조로써 알 수 있듯이 하나의 목표를 향해 가게 되면 다른 목표에서 멀어지는 움직임이 발생한다. 그러나 마지막에는 결국 최적성이라는 목표에 가까워진다. 극단적으로 말해 두 프로세스는 직접적으로 최적성을 달성하려고 노력하지 않더라도 최적성이라는 전반적인 목표를 달성하게 된다.
