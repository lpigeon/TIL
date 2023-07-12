# Chapter 13: Policy Gradient Methods  

지금까지 배워왔던 거의 모든 방법은 행동 가치 방법이었다. 행동 가치 방법은 행동의 가치를 학습하고 추정된 행동 가치에 기반하여 행동을 하는 방법이며 따라서 행동 가치 추정 없이는 정책을 사용할 수 없었다. 이번 챕터는 가치함수를 추정하지 않고 행동을 선택할 수 있는 방법을 배우는데 이것이 parameterized policy이다.  
parameterized policy의 파라미터 벡터를 $\theta$ 라고 하면 다음과 같이 나타낼 수 있다.  

![parameterized_policy](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/parameterized_policy.png?raw=true)  

만약, 학습된 가치 함수도 사용하는 방법이라면 가치 함수의 가중치 벡터를 $w$ 로 하고 $\hat{v}(s,w)$ 와 같이 표시할 수 있다.  

정책의 성능을 측정하는 scalar 값인 $J(\theta)$ 값을 정의하고 gradient로 파라미터 $\theta$를 업데이트하는 것을 나타내면 다음과 같다. 이러한 업데이트 방식은 gradient 상승을 근사화 한다.  

![theta_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/theta_update.png?raw=true)  

이러한 방식을 policy gradient methods라고 한다. 엊책과 가치 함수를 모드 근사화 하는 방법은 actor-critic methods라고 불리며 actor는 학습된 정책을 참조하는 것이고, critic은 주로 상태 가치 함수에서 학습된 가치 함수를 참조한다.  

## 13.1 Policy Approximation and its Advantages  

policy gradient 방법에서 정책 $\pi(a|s, \theta)$ 이 파라미터 벡터 $\theta$ 로 미분이 가능하다면 즉, ∇$\pi(a|s, \theta)$ 가 존재한다면 어떤 형태로든 정책을 파라미터화 할 수 있다. 실제로는, 탐색을 보장하기 위해 정책이 deterministic되는 일이 없어야한다는 조건이 추가된다.  

만약, 행동 공간(action space)이 이산적이고 너무 크지 않다면, 상태 행동 쌍마다 파라미터화된 숫자적인 선호도 $h(s, a, \theta)$ 를 나타낼 수 있다. 각 상태에서 가장 높은 선호도를 가진 행동들은 선택될 확률이 가장 높게 부여되고 exponential soft-max 분포에 따라 선택될 수 있다.  

![policy_h](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/policy_h.png?raw=true)  

분모는 각 상태에서의 행동 확률의 합이 1이 되도록 하기 위해 필요한 부분이다. 이러한 정책 파라미터화 방법을 $soft-max$ $in$ $action$ $preferences$라고 부른다.  

행동 선호도는 파라미터화가 된다면 인공 신경망을 사용하든 아래와 같은 간단한 선형 함수를 사용해도 된다.  

![linear_h](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/linear_h.png?raw=true)  

행동 선호도에 soft-max를 적용하는 방식의 장점 중 하나는 근사된 정책이 deterministic 정책에 가까워질 수 있다는 것이다. 이전의 $\epsilon$-greedy 정책 같은 경우에는 수렴하더라도 $\epsilon$ 의 확률로 랜덤한 행동을 하는 부분이 남아 있어야만 했다.  
또 다른 장점으로는 임의의 확률로 행동을 선택할 수 있다는 것이다. 함수 근사화가 중요한 문제에서는 최적의 근사 정책이 확률적(stochastic)일 수 있다. 행동 가치 방법은 이러한 확률적인 최적 정책을 찾는 자연스러운 방법이 없지만, 정책 근사 방법은 가능핟.  
정책 파라미터화(policy parameterization) 방법이 행동 가치를 파라미터화(action-value parameterizatio)하는 것보다 더 좋은 점은 더 간단한 함수라는 것이다. 일부 문제에서는 행동 가치 함수가 더 간단하겠지만 정책 파라미터화 방법이 일반적으로 더 빨리 학습하고 우수한 수렴 정책을 얻게 된다.  
마지막으로, 정책 파라미터화의 선택은 때로는 정책의 원하는 형태에 대한 사전 지식을 강화 학습 시스템에 주입하는 좋은 방법일 수 있다는 것이다.  

## 13.2 The Policy Gradient Theorem  

정책 파라미터화는 이론적으로도 장점이 있는데 $\epsilon$-greedy 정책의 경우 행동 가치 함수의 추정값이 변함에 따라 각 행동에 배정되는 확률이 급격히 변할 수 있다는 부분이 있지만 정책 파라미터화는 매끄럽게 변화한다. 이러한 이유로 인해 policy gradient 방법은 행동 가치 방법 보다 더 강력한 수렴을 보장한다.  
  
episodic한 형태와 continuing한 형태에서의 성능 측정 $J(\theta)$를 다르게 정의하는데 이번 장에서는 episodic한 형태를 다룬다.  
에피소드의 성능 측정을 에피소드의 시작 상태의 가치로 정의하고 모든 에피소드가 특정한(랜덤이 아닌) 시작 상태 $s_0$에서 시작된다고 가정한다. 그러면 $J(\theta)$를 다음과 같이 나타낼 수 있다.  

![episodic_h](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/episodic_h.png?raw=true)  

위 식에서 $v_{\pi_\theta}$는 $\theta$ 에 의해 결정된 정책 $\pi_\theta$ 의 실제 값이다. 이후에서는 할인율 $\gamma$ 을 고려하지 않는다고 가정한다($\gamma$ = 1).  
  
함수 근사를 할 때, $J$ 값이 개선되는 것을 보장하는 방향으로 파라미터 $\theta$ 를 움직이는 것은 쉬운 일은 아니다. 성능이 행동 선택과 해당 선택이 이루어지는 상태 분포에 모두 의존하며, 이 두 가지가 정책 파라미터에 영향을 받기 때문이다. 특정한 상태가 주어졌을 때, 정책 파라미터의 행동 및 보상에 대한 효과는 파라미터화에 대한 지식을 기반으로 비교적 간단하게 계산될 수 있지만 정책이 상태 분포에 미치는 효과는 환경에 따라 달라지며 일반적으로 알려져 있지 않다.  
  
gradient가 우리가 알지 못하는 상태 분포의 변화에 영향을 받을 때 이를 추정하는 방법은 $policy$ $gradient$ $theorem$이라는 정리를 이용해서 가능하다.  

![policy_gradient_theorem](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/policy_gradient_theorem.png?raw=true)  

여기서 $\mu$ 는 정책 $\pi$ 하의 on-policy distribution이다. 등식 대신 비례 기호가 쓰인 이유는 episodic 문제의 경우 에피소드의 평균 길이만큼 비례하고, continuing case는 1대1로 비례해서 등식이 성립하기 때문이다. 아래는 episodic 문제에 대한 도출 과정이다.  

![proof_policy_gradient_theorem_epicodic](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/proof_policy_gradient_theorem_epicodic.png?raw=true)  

## 13.3 REINFORCE: Monte Carlo Policy Gradient  

stochastic gradient ascent(SGD) 방법에서는 한 샘플의 gradient의 기대값이 실제 성능 $J$ 의 gradient의 기대값이 같으면 된다. 샘플 gradient는 gradient에 비례하기만 하면 되며, 비례 상수는 임의로 선택할 수 잇는 step 크기 $\alpha$ 로 둘 수 있다.  

![sgd_j](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/sgd_j.png?raw=true)  

![sgd_theta](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/sgd_theta.png?raw=true)  

위 식의 우변의 $\mu$ 는, 현재 따르고 있는 타겟 정책 $\pi$ 하에서 각 상태(state)들이 얼마나 방문되느냐를 말해주고, 이를 가중평균한 것은 결국 $\pi$ 하에서의 기대값과 같다. 이때 위 식만 사요하여 SGD를 업데이트 하는것을 $all-actions$ $method$ 라고 한다.  

이 장에서는 전통적인 REINFORCE 알고리즘에 집중하며 이 알고리즘은 시간 $t$ 에서 실제로 수행된 단 하나의 행동 $A_t$ 만을 업데이트에 포함한다. 아까의 식에서, $\pi(a|S_t, \theta)$ 를 분모와 분자에 동일하게 곱해주면 아래의 첫번째 식이 된다. 아까전의 $\mu$ 를 이용한 것과 마찬가지로, $\pi(a|S_t, \theta)$ 는 행동 $a$ 들의 분포를 나타내주므로, 어떤 특정한 행동 $A_t$ 을 샘플하는 것과 기대값이 동일하게 된다.  

![reinforce_j_gradient](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/reinforce_j_gradient.png?raw=true)  

그렇다면 REINFORCE 알고리즘의 업데이트는 다음과 같다.  

![theta_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/theta_update.png?raw=true)  

이 업데이트 식은 매우 직관적인데, gradient부분은 결국 업데이트가 해당 행동 $A_t$ 를 선택할 확률을 높이는 쪽으로 움직이는 것을 말하고, 수익(return) $G_t$ 가 높을수록 이 업데이트의 크기는 커지며, 현재의 확률인 분모값이 클수록 업데이트의 크기는 작아진다. 전자는 가장 높은 반환을 가져다주는 행동을 선호하는 방향으로 파라미터를 가장 많이 움직이게 하므로 의미가 있으며 후자는 자주 선택되는 행동이 우위에 있는 경우 가장 높은 반환을 내지 않더라도 이길 수 있기 때문에 의미가 있다.  
  
REINFORCE에서는 시간 $t$ 부터 에피소드의 끝까지의 모든 미래 보상을 포함한 완전한 반환값을 사용하므로 MC 알고리즘이라고 할 수 있다.  

![mc_policy_gradient](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/mc_policy_gradient.png?raw=true)  

위 REINFORCE 알고리즘의 pseudocode에서 한가지 차이점은 마지막 부분에 ∇$\ln \pi(A_t|s_t,\theta)$ 이다. 이는 단순히  ∇$\ln x$ = ∇$x$ / $x$ 의 공식을 치환해서 사용한 것이다. 이때, ∇$\ln \pi(A_t|s_t,\theta)$ 부분의 벡터를 $eligibility$ $vector$ 라고 부르며 policy parameterization이 등장하는 유일한 부분이다. 또 한가지 차이점은 $\gamma$ 값인데, 위에서 알고리즘 설명할 때는 non-discounted case 즉 $\gamma$=1을 가정했지만, 위 의사코드에서는 이를 일반화한 것이기 때문이다.

## 13.4 REINFORCE with Baseline  

policy gradient theorem에서는 다음과 같이 임의의 $baseline$ $b(s)$와 행동 가치의 비교를 포함해서 일반화 할 수 있다.  

![generalized_j](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/generalized_j.png?raw=true)  

이 $baseline$은 a와 함께 움직이지 않는 한 한 어떤 함수여도 되며, random variable이어도 된다. 그 이유는 다음의 식에서 확인할 수 있다.  

![b](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/b.png?raw=true)  

$baseline$을 포한한 일반적인 REINFORCE 알고리즘의 업데이트 식은 다음과 같다.  

![generalized_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/generalized_update.png?raw=true)  

일반적으로, $baseline$은 업데이트의 기대값을 변경하지 않지만, 분산에 큰 영향을 미칠 수 있다. 일부 상태에서는 모든 행동이 높은 가치를 가지고 있으므로 높은 $baseline$이 필요하며, 덜 가치 있는 행동과 높은 가치를 가진 행동을 구별해야 한다. $baseline$ 선택 중에 자연스러운 값은, 상태 가치(state value)인 $\hat{v}(S_t, w)$를 사용하는 것이다. REINFORCE가 MC 방법이므로 상태 가치 함수의 파라미터 $w$ 또한 MC 방식으로 추정하는 것이 자연스러울 것이다.  

![reinforce_with_baseline](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/reinforce_with_baseline.png?raw=true)  

이 알고리즘은 $\alpha^\theta$ 와 $\alpha^w$ 두 개의 step size 파라미터를 가지고 있다. $\alpha^w$ 를 정하는 것은 상대적으로 쉬우며 그 값은 다음과 같이 설정할 수 있다.  

![alpha_w](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/alpha_w.png?raw=true)  

$\alpha^\theta$ 의 경우에는 보상의 변동 범위와 정책 파라미터화에 따라 최적의 값이 달라지기 때문에 값을 정하는 것이 명확하지 않다.  
  
다음은 REINFORCE 알고리즘과 REINFORCE with baseline 알고리즘의 성능 비교 그래프이다. 이때 $baseline$으로 사용된 근사 상태 가치 함수는 $\hat{v}(S_t, w)$ = $w$ 이다.  

![comparison_reinforce](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/comparison_reinforce.png?raw=true)  

## 13.5 Actor–Critic Methods  

앞선 REINFORCE with baseline은 학습된 상태 가치 함수가 각 상태 전이의 첫번째 상태의 가치만을 추정하기 때문에 actor-critic 방법에 속하지 않는다. actor-critic 방법에서는 상태 가치 함수가 두번째 상태에도 영향을 준다. 할인(discount)된 두 번째 상태의 추정 가치와 보상을 더한 것은 단일 step 반환인 $G_{t:t+1}$ 이 된다. 이런 단일 step 반환은 앞선 TD 방법에서 볼 수 있다. TD 방법은 MC, REINFORCE, REINFORCE with baseline와는 다르게 편향(bias)이 생기지만 분산과 계산적 측면에서 더 우수하다.  

먼저, one-step actor–critic 방법을 살펴본다. one-step의 장점은 완전히 온라인 및 점진적이며, 동시에 eligibility traces의 복잡성을 피할 수 있다는 것이다. REINFORCE에서 사용하는 full return을 one-step return으로 bootstrap하면 다음과 같다.  

![bootstrap_update](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/bootstrap_update.png?raw=true)  

이러한 업데이트와 함께 사용할 수 있는 자연스러운 상태 가치 함수 학습 방법은 semi-gradient TD(0)이다.  

![one_step_actor_critic](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/one_step_actor_critic.png?raw=true)  

이제 이 one-step 알고리즘을 n-step 방법으로 확장하고, 더 나아가 $\lambda$-return으로 일반화하는 건 간단하다. One-step return을 각각 $G_{t:t+n}$과 $G^{\lambda}~{t}$로 대체하면 된다. Backward view λ-return으로 확장하는 것도 actor와 critic 각각 다른 eligibility trace를 사용하면 된다.  

![actor_critic_with_eligibility_traces_episodic](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/actor_critic_with_eligibility_traces_episodic.png?raw=true)  

## 13.6 Policy Gradient for Continuing Problems  

episodic한 문제를 continuing한 문제로 확장하는 것은 성능 $J$을 시간 단계 당 평균 보상률로의 정의를 필요로 한다.  

![continuing_j](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/continuing_j.png?raw=true)  

여기서 $\mu$ 는 정책 $\pi$ 하의 steady-state distribution이며 다음과 같이 나타낼 수 있다.  

![continuing_mu](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/continuing_mu.png?raw=true)  

이러한  $\mu$ 는 존재하여야 하며 $S_0$와 독립이어야 한다(ergodicity assumption). 이는 정책 $\pi$ 에 따라 행동하면, 동일한 분포가 유지되는 특별한 분포이다.  

![special_distribution](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/special_distribution.png?raw=true)  

다음은 continuing한 문제에서의 actor-critic 알고리즘이다.  

![actor_critic_with_eligibility_traces_continuing](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/actor_critic_with_eligibility_traces_continuing.png?raw=true)  

continuing한 문제에서는 다음과 같이 각 값을 정할 수 있다.  

![continuing_v_q_g](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/continuing_v_q_g.png?raw=true)  

위 정의를 이용해 policy gradient theorem을 증명하는 방법은 아래와 같다.  

![proof_policy_gradient_theorem_continuing](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter13/proof_policy_gradient_theorem_continuing.png?raw=true)  