# Chapter 3: Finite Markov Decision Processes

## 3.1 The Agent-Environment Interface  

해결할려는 문제 => 각 상태 s에서 각 행동 a의 가치 $q_*(s,a)$가 주어졌을때 각 상태 $v_*(s)$의 가치를 추정하는 것.  
앞선 bandit problem과 다른 것은 bandit problem은 각 행동 a의 가치를 추정함 -> $q_*(a)$.

### Markov Decision Processes(MDP)

![mdp](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/mdp.png?raw=true)  

Markov Decision Processes는 의사결정 과정을 모델링 하는 수학적인 틀을 제공하는 일종의 과정이다.  
위에 그림은 agent 환경에서의 MDP 그림이며 이산(discrete)적인 시간마다 아래와 같은 trajectory를 생성하게 된다.  

![trajectory](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/trajectory.png?raw=true)  

$S_0$이라는 환경에서 $A_0$라는 행동을 했을 때, $R_1$이라는 보상을 받게 되는 것이다. 단, reward인 R에 경우 앞선 S와 A에 의해 주어지는 보상이기
때문에 시간 t가 한번 앞선다.  

- finite MDP
만약, S, A, R이 유한한(finite) 개수를 가진다면 $R_t$와 $S_t$는 이전 이전 S와 A에만 의존하고 다음과 같은 잘 정돈된 이산확률분포를 가지게 된다.  

![finite_MDP](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/finite_MDP.png?raw=true)

![finite_MDP_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/finite_MDP_2.png?raw=true)  

$p$는 dynamics of MDP라고 하며 [0,1]사이의 값이다. 위 식으로써 알 수 있는 것은 어떠한 state, action을 알기 위해선 바로 전 state와 action만 필요하다는 것이다. 이러한 식은 여러가지로 변형 가능한데 다음과 같다.  

![finite_MDP_3](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/finite_MDP_3.png?raw=true)  
![finite_MDP_4](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/finite_MDP_4.png?raw=true)  
![finite_MDP_5](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/finite_MDP_5.png?raw=true)  

- MDP 이해를 위한 그림  
![recycling_robot](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/recycling_robot.png?raw=true)  

## 3.2 Goals and Rewards

강화학습의 목표는 agent가 특정한 목적 또는 목표를 달성했을 때 얻는 reward를 최대화 하는것이다. 이때 reward의 최대화는 즉각적인것이 아닌 장기적으로 누적된 보상을 의미한다. 이것을 reward hypothesis(보상 가설)라고 한다.  

- reward hypothesis : That all of what we mean by goals and purposes can be well thought of as
the maximization of the expected value of the cumulative sum of a received
scalar signal (called reward).  

이때 주의할 점은 행동에 대한 보상을 실제로 우리가 달성하길 원하는 것을 정확히 설정해야 한다는 것이다. 예를 들어, 체스에서 승리했을때 +1을 줘야지 상대말을 잡았을 때 +1을 주면 안된다는것이다. 후자에 경우는 체스에서 지더라도 상대방의 말을 잡는 방식으로 학습이 될 수도 있기 때문이다.  

## 3.3 Returns and Episodes

앞선 section에서 보상의 누적된 최대화가 필요하다고 했다. 우리는 누적된 보상을 최대화하기 위해서 기대값의 최대화가 필요하다. 아래의 식은 보상을 순서로한 특정한 함수 G의 대한 정의이다.  

![G_t](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/G_t.png?raw=true)

- $episodic$ $task$  

이때, T는 최종 시간단계를 의미한다. 여기서는 episode(에피소드)라는 개념이 나오는데 에피소드란 에이전트 - 환경의 독립적인 상호작용이라고 이해할 수 있다. 이러한 접근 방식은 한 에피소드가 어떻게 끝났는 다음 에피소드는 이전 에피소드와는 독립적이게 시작된다는 것이 중요하다. 따라서 모든 에피소드는 서로 다른 결과에 대한 서로 다른 보상이 있는 상태로 종료(terminal state)되게 된다. 이러한 작업을 $episodic$ $task$라고 한다.  

- $continuing$ $tasks$

그렇다면 T가 무한하게 커진다면 어떻한 상황이 될까? 이러한 상황은 에피소드가 자연스럽게 분리되지 않고 제한없이 진행하는 상황이다. 이러한 작업을 $continuing$ $tasks$ 라고 한다. 이러한 작업은 위 식에 문제가 생기는데, T = $\infty$이므로 보상이 무한대가 될 수 있기 때문이다. 이러한 문제점을 해결하고자 discounting(할인)이라는 개념이 나온다. 이러한 접근방식은 미래의 보상을 할인률을 거쳐 얻고 이러한 보상을 최대화 하는 행동을 하게 하는것이며 다음과 같은 식을 따른다.  

![discounting_G_t](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/discounting_G_t.png?raw=true)  

여기서 $\gamma$는 discount rate(할인율)이라고 불리는 매개변수이며 [0, 1]의 값을 가진다. 이러한 할인율은 미래의 보상의 현재 가치를 가르킨다. 만약, $\gamma$가 0이라면 $myopic$(근시안적)인 상태라고 하며 즉각적인 보상을 최대화할려고 한다. 반대로 $\gamma$이 1에 가까울 수록 미래 보상을 더 강하게 고려한다.  
  
위 식을 다음과 같이 변형할 수 있다.  

![discounting_G_t_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/discounting_G_t_2.png?raw=true)  

이러한 식은 보상 시퀀스로부터 반환값을 더 쉽게 계산해준다.

## 3.4 Unified Notation for Episodic and Continuing Tasks  

앞선 section에서 $episodic$ $task$와 $continuing$ $tasks$에 대해서 다루었지만 각각을 동시에 다루는 표기법을 사용하는 편이 더 유용하다. 각 에피소드 시간 단계를 0에서 부터 번호를 붙이고 $S_t$ 뿐만 아니라 에피소드 번호를 표시하기 위해 $S_t,i$라는 표현을 고려해 볼 수 있다. 하지만 $episodic$ $task$ 에서는 에피소드를 구분할 필요가 없으므로 에피소드 번호를 명시적으로 생략하여 $S_t$라고 표기한다.  

![state_transition_diagram](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/state_transition_diagram.png?raw=true)  

또한, $episodic$ $task$, $continuing$ $tasks$ 의 반환값(보상의 합)을 단일 표기법을 위해 위와 같은 상태 전이 다이어그램을 생각해 볼 수 있다. 에피소드가 끝나는 마지막 사각형에서는 자기자신에게 무한히 돌아오는 에피소드가 진행된다. 이러한 방식으로 반환값을 구하면 다음과 같다.

![discounting_G_t_3](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/discounting_G_t_3.png?raw=true)  

## 3.5 Policies and Value Functions  

거의 모든 강화학습 알고지름은 value function(가치 함수)의 추정이 가장 중요하다. 가치 함수는 state 또는 state-action의 함수로써, state가 "얼마나 좋은지" 또는 state에서 action이 "얼마나 좋은지"를 추정하는 것이다. 이러한 가치 함수를 $policy$(정책)이라는 방식에 따라 정의된다. 정책의 공식적인 정의는 다음과 같다.

- $policy$ : a mapping from states to probabilities of selecting each possible action  

즉, 어떤 state에서 가능한 action을 선택할 확률에 대한 매핑이다. agent가 정책을 따른다면 시간 t에서의 정책 $\pi$는 $S_t$=S에서 $A_t$=a를 선택할 확률 $\pi(a|s)$라고 한다.  

다음은 정책 $\pi$를 따르는 두가지의 가치함수이다.  

- state-value function  
![state_value_function](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/state_value_function.png?raw=true)  

- action-value function  
![action_value_function](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/action_value_function.png?raw=true)  

위 두 가치함수는 experience(경험)으로 부터 추정될 수 있는데 만약 agent가 정책을 따르고 각 상태에서의 실제 반환값의 평균을 유지한다면 해당상태를 만난 횟수가 무한에 접근 할 때, 상태 가치인 $v_\pi$(s)에 수렴할 것이다. 또한, 각 상태에서의 행동을 유지한다면 행동 가치인 $q_\pi$(s,a)로 수렴할 것이다. 이러한 추정 방식을 Monte Carlo methods라고 한다. Monte Carlo 방법이란 실제 반환값의 많은 무작위 샘플을 평균화 하는것을 말한다.  
  
가치 함수를 추정하는 방법에는 dynamic programming적인 방식이 있는데 이 방식은 가치함수의 재귀적인 관계를 이용하는 방식이다. 재귀적인 관계는 Bellman equation을 통해 알아 낼 수 있다. Bellman equation이란, 현재 state/action 가치과 후속 state/action 가치 사이의 관계를 나타내는 방정식이다. 다음은 $v_\pi$의 Bellman equation 표현법이다.  
![bellman_eq_v](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/bellman_eq_v.png?raw=true)

- Bellman equation를 통해 state-value function 과 action-value function과의 관계 확인

state value function는 다음과 같이 변형할 수 있다.  
![state_value_function_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/state_value_function_2.png?raw=true)

①은 t+1 시점에 받는 즉각적인 보상이며 ②는 할인율이 곱해진 미래의 보상이다. 이 식을 그림으로 나타내면 다음과 같다.  
![state_value_function_3](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/state_value_function_3.png?raw=true)

이때, next state value function과의 관계를 살펴보기 위해 $q_\pi(s,a)$를 다음과 같이 표현할 수 있다.
![aciton_value_function_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/aciton_value_function_2.png?raw=true)

위 식을 합치면 최종적으로 $v_\pi(s)$의 식은 다음과 같이 나온다.  
![bellman_eq_v_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/bellman_eq_v_2.png?raw=true)  

위 글의 참고와 action value function의 Bellman equation은 다음의 사이트에서 확인 가능하다.  
[https://sumniya.tistory.com/5](https://sumniya.tistory.com/5)  

## 3.6 Optimal Policies and Optimal Value Functions  

$optimal$ $policy$란 최소한 하나의 정책은 다른 모든 정책보다 우수하거나 동일하다는 것이다. 이러한 optimal policy는 $\pi_*$라고 표기하며 ($\pi_*$가 두개 이상일 수도 있다.) 동일한 상태 가치 함수를 공유한다. 그러한 상태 가치 함수를 $optimal$ $state-value$ $function$이라고 하며 $v_*$라고 표기한다. 또한, optimal policy는 동일한 행동 가치 함수를 공유하며 그러한 행동 가치 함수를 $optimal$ $action-value$ $function$이라고 하며 $q_*$라고 표기한다. 다음은 $v_*$ 와 $q_*$ 정의이다.  

![optimal_state_value_function](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/optimal_state_value_function.png?raw=true)  
![optimal_action_value_function](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/optimal_action_value_function.png?raw=true)  

$q_*$를 $v_*$를 사용하여 표현하면 다음과 같다.  
![q_star](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/q_star.png?raw=true)  

$optimal$ $policy$를 따르는 Bellman equation을 Bellman Optimality Equation이라고 한다. 다음은 $v_*$ 와 $q_*$의 Bellman Optimality Equation이다.  

![v_bellman_optimality_equation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/v_bellman_optimality_equation.png?raw=true)  
![q_bellman_optimality_equation](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter3/q_bellman_optimality_equation.png?raw=true)  

유한한 MDP에 대해서 $v_*$의 Bellman Optimality Equation은 유일한 해를 가지고 있다고 한다. 즉, 각 상태에 대한 하나의 방정식이 주어진 방정식 시스템이다. 환경의 dynamics $p$를 알고 있다면, 원칙적으로 $v_*$와 $q_*$비선형 방정식을 해결할 수 있다.  

### one-step search  

$v_*$를 구한 후에는 비교적 쉽게 $optimal$ $policy$를 구할 수 있다. 어떠한 s에 상황에서 다음 하나 이상의 aciotn의 Bellman Optimality Equation의 최댓값을 얻을 수 있을 것이다. 이러한 action에만 0 이상의 확률을 할당하는 최적 정책을 취하면 이러한 것을 one-step search라고 한다. 우리는 이러한 action만 찾아가도(greedy) 최적 정책을 찾을 수 있다. 왜냐하면 Bellman Optimality Equation에 의해 $v_*$이후에 일어날 모든 action의 보상결과가 이미 반영되어 있기 때문이다.  

### Bellman Optimality Equation 풀이 가능성  

Bellman Optimality Equation만 찾으면 강화학습문제를 푼것이나 다름없다. 하지만 Bellman Optimality Equation은 직접적으론 유용하지 않다. Bellman Optimality Equation을 풀기 위해선 다음과 같은 세가지 가정에 의존한다고 한다.  

1. 환경의 동적인 정보가 정확하게 알려져 있다는 가정
2. 계산 리소스가 계산을 완료하기에 충분하다는 가정
3. 상태가 MDP를 따른다는 가성  

위 세가지를 모두 만족하는 경우는 거의 없으므로 현실적으로 Bellman Optimality Equation의 풀이는 불가능에 가까우며 일반적으로 근사 해법을 사용한다.  