# Chapter 5: Monte Carlo Methods  

DP를 사용한 학습방법의 단점은 model을 완벽하게 알아야 하는 것(planning)에 있다. 이러한 단점을 보완하고자 model을 정확히 알지 못해도 경험을 통해 학습할 수 있는 것(learning)이 Monte Carlo Methods 이다.  
Monte Carlo Methods은 샘플 반환값의 평균을 기반으로 강화 학습 문제를 해결하는 방법이다. 또한, 명확한 반환값을 사용할 수 있도록 에피소드형 작업에서만 정의되며 각 에피소드가 완료된 후에만 가치 추정치와 정책이 변경된다.  
반환값의 평균을 사용한다는 관점에서 앞선 챕터의 bandit 알고리즘과 유사하지만 차이점은 bandit처럼 하나의 상태가 아니라 여러상태가 있으며 각 상태에서 취한 행동에 따라 반환값이 달라진다는 것이다.

## 5.1 Monte Carlo Prediction  

정책 $\pi$를 따르는 상태 s의 가치함수인 $v_\pi(s)$를 추청한다고 가정해보자. 각 에피소드에서 상태 s의 각 발생을 s를 방문한 것으로 간주한다. 에피소드에서 s가 처음 반문되는 것을 첫 방문(first visit)라고 한다. 이 방문에 따라 MC는 두가지 방식으로 나누어진다.

- First-Visit Monte Carlo : 상태 S에 대한 첫 방문 이후 반환값의 평균으로 $v_\pi(s)$를 추정하는 방식  
- Every-Visit Monte Carlo : 상태 S에 대한 모든 방문 이후 반환값의 평균으로 $v_\pi(s)$를 추정하는 방식  

각 방식은 매우 유사하지만 약간 다른 이론적 특성을 가지고 있다. 본 챕터에서는 First-Visit Monte Carlo 방식을 중점적으로 살펴보며 Every-Visit Monte Carlo 방식은 9장, 12장에서 논의된다.  

두 방식 모두 s에 대한 방문 횟수가 무한대로 갈 때 $v_\pi(s)$로 수렴한다. First-Visit Monte Carlo은 각 반환값이 독립적이고 동일하게 $v_\pi(s)$의 유한한 분산을 가진 추정치이기 때문에 큰수에 법칙에 따라 추정치의 평균은 기대값으로 수렴한다. Every-Visit Monte Carlo는 덜 직관적이지만 마찬가지로 수렴한다.  

다음은 First-Visit Monte Carlo의 의사 코드이다.  

![First_visit_MC](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/First_visit_MC.png?raw=true)  

MC와 DP와의 차이점중 하나는 MC는 각 상태에 대한 추정치가 독립적이라는 것이다. 한 상태에 대한 추정치는 DP와 달리 다른 상태의 추정치에 의존하지 않는다. 즉, MC는 이전 장에서 정의한 부트스트랩을 수행하지 않는다. 이러한 독립성 덕분에 MC는 필요한 상태를 제외한 모든 다른 상태를 무시할 수 있다는 강력한 장점을 가진다.  

## 5.2 Monte Carlo Estimation of Action Values  

MC는 model을 정확히 모를때도 사용할 수 있다고 했다. 좀 더 자세하게 알아보면 model이 없는 경우에는 상태값보다 행동값(상태-행동 쌍의 값)을 추정하는 것이 유용하다. model이 없다면 다음의 상태값을 확인하여 가장 좋은 행동만 찾아가는 방식(DP)을 사용할 수 없으므로 각 행동에 대한 값을 추정해야 한다. 따라서, MC의 주요 목표는 $q_*$를 추정하는 것이다.  
  
상태-행동 쌍 s, a는 해당 상태 s가 방문되고 해당 행동 a가 선택된 경우에만 해당 에피소드에서 방문된 것으로 간주된다. Every-Visit Monte Carlo의 경우는 모든 방문을 마치고 반환값을 추정하고 First-Visit Monte Carlo는 각 에피소드에서 행동이 선택된 후 반환값을 평균화 한다.  

방문이 무한히 많으면 수렴은 반드시 일어나겠지만 문제점이 있을 수 있다. 바로 많은 상태-행동 쌍이 결코 방문되지 않을 수도 있다는 것이다. 만약, 정책 $\pi$가 deterministic하다면, $\pi$를 따를 때 각 상태에서 하나의 행동에 대한 반환값만 관찰할 수 있다. 평균화할 반환값이 없다면, 다른 행동에 대한 MC 추정이 개선되지 않을 것이다.  
  
이러한 문제점을 해결하고자 챕터2에서의 k-armed bandit 문제를 떠올려보자. 단순한 탐욕적 행동을 막고자 확률로서 탐색을 보장하는 방법을 사용했다. MC에도 마찬가지로 에피소드가 상태-행동 쌍에서 시작되고, 각 쌍이 시작으로 선택될 확률이 0이 아닌 값을 갖도록 지정하는 방법을 사용하는 것이다. 이것을 "탐색 시작 가정(assumption of exploring starts)" 이라고 한다.  

## 5.3 Monte Carlo Control  

앞선 챕터에서는 DP를 사용하여 policy iteration를 하는 방식을 설명했다.(GPI) 이번 장에서는 MC를 사용한 policy iteration를 다룬다. 다음의 그림은 MC에서의 GPI를 나타낸다.  

![MC_GPI](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/MC_GPI.png?raw=true)  

주목해야할 부분은 DP와는 다르게 상태가치함수가 아니라 행동가치함수를 사용하여 평가와 개선이 이루어진다는 것이다.  

![MC_policy_iteration](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/MC_policy_iteration.png?raw=true)  

위 그림과 같이 많은 에피소드를 경험하며 근사 행동 가치 함수가 점진적으로 실제 함수에 근접해 간다. 에피소드는 탐색적 시작으로 생성된다고 가정한다. 이러한 가정하에 MC 방법은 임의의 $\pi_k$에 대해서 $q_{\pi_k}$를 정확하게 계산한다.  
정책 개선은 현재 값 함수에 대해 탐욕적인 정책을 만들어내는 것으로 수행된다. 이경우, 행동 가치 함수가 주어지므로 탐욕적인 정책을 구성하기 위해 model이 필요하지 않다.  

![greedy_policy](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/greedy_policy.png?raw=true)  

다음의 그림을 살펴보면 탐욕적 정책이 다음 정책보다 더 나은정책이거나 동등한 정책임을 보장하는지 알 수 있으며 이는 전체적인 과정이 최적 정책과 최적 값 함수로 수렴함을 보장해준다.  

![greedy_policy_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/greedy_policy_2.png?raw=true)  

MC 방법의 수렴성을 보장하기 위해 두 가지 비현실적인 가정을 하였는데 다음과 같다.  

1. 에피소드가 탐색적 시작(exploring starts)을 가진다.
2. 정책 평가를 무한한 에피소드로 수행 할 수 있다.  

이러한 가정은 실용적인 알고리즘을 얻기 위해 모두 제거해야 한다. 첫번째 가정은 이 챕터의 후반부에서 다룬다. 두번째 가정에 대해서 초첨을 맞춰보자.  
  
두번째 가정은 고전적인 DP방법에서도 발생하는데 이것을 해결하기 위한 방법은 두가지가 있다.  

하나는 정책 평가에서 $q_{\pi_k}$를 근사하는 개념을 유지하는 것이다. 추정치의 오차의 크기와 확률을 얻기 위해 측정과 가정을 수행하고, 이러한 오차가 충분히 작아지도록 충분한 단계를 정책 평가 과정에서 수행한다. 이 접근 방식은 근사 수준까지 올바른 수렴을 보장하는 면에서 완전히 만족스러울 수 있지만 실제로는 작은 문제를 제외하고는 사용하기에는 에피소드가 너무 많이 필요할 것이다.  

두번째는 정책 평가를 완료하기 전에 정책 개선으로 돌아가려는 시도를 포기하는 것이다. 이러한 방식의 극단적인 형태가 앞선 챕터에서 설명한  값 반복(value iteration)이다. 값 반복에서는 정책 개선 단계마다 반복적인 정책 평가를 한 번만 수행한다. 더 극단적인 방법으로는 인플레이스(in-place) 버전의 값 반복이 있다. 인플레이스 값 반복에서는 개선과 평가 단계를 단일 상태에 대해 번갈아 수행한다.  
  
MC 정책 반복에 있어서 에피소드 단위로 평가와 개선을 번갈아가며 수행하는 것이 자연스럽다. 이러한 방식으로 구성된 간단한 알고리즘을 Monte Carlo with Exploring Starts라고 부른다. 다음은 Monte Carlo with Exploring Starts의 의사 결정코드이다.  

![MC_ES](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/MC_ES.png?raw=true)  

MC ES에서는 초기 행동을 random하게 선택하며 각 상태-행동 쌍의 모든 리턴값을 누적하고 평균을 구한다. MC ES는 모든 리턴값을 사용하기 때문에 가치함수는 수렴한다. 하지만 초기 행동을 random하게 선택하지 못하는 상황에선 MC ES의 사용은 어려울 것이다.  

## 5.4 Monte Carlo Control without Exploring Starts  

Exploring Starts을 피하는 방법은 에이전트가 계속해서 그들을 선택하도록 하는 것이다. 이를 위한 두가지 접근 방식이 있다.  

- On-policy : 결정을 내리는 데 사용되는 정책을 평가하거나 개선시킴
- Off-policy : 데이터를 생성하는 데 사용되는 정책과는 다른 정책을 평가하거나 개선시킴  

앞서 설명한 MC ES는 On-policy 방법이다. 이번 장에서는 Exploring Starts를 사용하지않고도 On-policy를 사용한 MC 방법을 설명한다.  

On-policy 제어 방법에서는 일반적으로 정책이 soft하다고 말하는데 여기서 soft하다는 것은 다음과 같다.  

![soft](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/soft.png?raw=true)  

이 장에서는 On-policy 방법으로 $\epsilon$-greedy 정책을 사용한다. $\epsilon$-greedy는 $\epsilon$-soft 방식에 하나의 예이다.  

On-policy MC control의 전반적인 아이디어는 여전히 GPI이며 First-Visit MC를 사용하여 추정한다. 그러나 exploring starts 가정이 없기 때문에 현재 값 함수를 기반으로 정책을 탐욕적으로 개선하여도 비탐욕적인 행동의 추가 탐색이 방해되게 된다. 하지만 GPI에서는 정책을 반드시 탐욕적 정책으로 수렴시킬 필요가 없으며 탐욕적 정책으로만 이동시키면 된다. 다음은 On-policy First-Visit MC control에 대한 의사 결정 코드이다.  

![on_policy_first_visit_MC](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/on_policy_first_visit_MC.png?raw=true)  

## 5.5 Off-policy Prediction via Importance Sampling  

모든 학습은 최적행동을 하기 위해 모든 행동을 탐색하는 비-최적 행동을 해야한다. 이러한 딜레마를 해결하기 위해 앞선 장에서는 On-policy를 설명했다. 이번 장에서는 저 직관적인 접근 방식인 Off-policy에 대해 설명한다.  
  
Off-policy는 두가지 정책을 사용하여 학습한다. 하나는 학습 대상이 되는 정책으로서 최적 정책이 되고, 다른 하나는 더 많은 탐사를 수행하며 행동 생성에 사용되는 탐사 정책이다. 학습 대상이 되는 정책을 목표 정책(target policy)이라고 하고, 행동 생성에 사용되는 정책을 행동 정책(behavior policy)이라고 한다. 이러한 학습을 대상 정책이 아닌 데이터를 통해 이루어진다고 말하며, 전체 과정을 off-policy learning 이라고 한다.  
  
이 장에서는 목표 정책과 행동 정책이 고정된 상태에서 예측 문제를 고려하여 off-policy를 살펴본다. 즉, $v_\pi$ 또는 $q_\pi$를 추정하고자 하지만, 우리가 가지고 있는것은 poilcy $b$라는 다른 정책에 따라 생성된 에피소드들이다. 이러한 상태에서 목표 정책은 $\pi$이며 행동 정책이 $b$가 된다.  

$\pi$ 값을 추정하기 위해 $b$에서 생성된 에피소드를 사용하기 위해서는 $\pi$에 따라 취하는 모든 행동이 적어도 가끔은 $b$에서도 취해야 한다. 즉, $\pi$(a|s) > 0이라면 $b$(a|s) > 0임을 요구합니다. 이를 커버리지 가정(assumption of coverage)이라고 한다.  

거의 모든 Off-policy 방법은 중요도 샘플링(importance sampling)을 활용한다. 중요도 샘플링은 한 분포에서의 샘플을 통해 다른 분포에서의 기댓값을 추정하는 일반적인 기법이다. 우리는 중요도 샘플링을 오프-정책 학습에 적용하여 반환값을 목표 정책과 행동 정책에 따른 상대적인 경로 발생 확률에 따라 가중치를 부여한다. 이를 중요도 샘플링 비율(importance-sampling ratio)이라고 한다.  

![importance-sampling_ratio](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/importance-sampling_ratio.png?raw=true)  

중요도 샘플링 비율을 상대 확률로 나타내면 다음과 같다.(이때, $p$는 상태 전이 확률 함수이다.)  

![importance-sampling_ratio_p](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/importance-sampling_ratio_p.png?raw=true)  

비록 경로 확률은 MDP의 전이 확률에 의존하지만, 이들은 분자와 분모에 동일하게 나타나므로 상쇄된다. 중요도 샘플링 비율은 결과적으로 MDP에 의존하지 않고 두 정책과 시퀀스에만 의존하게 된다.  

현재 우리의 목표는 목표 정책에 따른 예상 반환값을 추정하는 것이다. 하지만 우리가 가지고 있는 것은 행동 정책으로 인한 반환값인 $G_t$뿐이다. $v_\pi(s)$를 얻기 위해 중요도 샘플링을 사용하여 다음과 같이 나타낼 수 있다.  

![v_pi](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/v_pi.png?raw=true)  

이제 정책 $b$를 따르는 고나찰된 에피소드 집합에서 반환값을 평균내어 $v_\pi(s)$를 추정하는 MC 방식을 사용할 수 있게 되었다. 이 때, 시간 단계를 에피소드 경계를 넘어가면서 증가하는 방식으로 번호를 매기는 것이 편리하다. 상태 S가 방문되는 모든 시간 간계의 집합을 $\tau(s)$라 정의 할 수 있다. First-Visit의 경우, s를 처음 방문한 시간 단계만 포함된다. 또한, $T(s)$는 시간 t 이후에 종료되는 첫 번째 시간을 나타내며, Gt는 t부터 $T(s)$까지의 반환값을 나타낸다.  

$v_\pi(s)$를 추정하기 위해 단순히 반환값을 비율로 조정하고 결과를 평균내면 다음과 같다.  
  
![v_s](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/v_s.png?raw=true)  

이와 같이 일반적인 평균으로 중요도 샘플링을 수행하는 경우, 이를 일반적인 중요도 샘플링(ordinary importance sampling)이라고 한다.
다른 대안으로는 가중 중요도 샘플링(weighted importance sampling)이 있다. 가중 중요도 샘플링은 가중 평균을 사용하여 정의되며 다음과 같다.  

![v_s_2](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter5/v_s_2.png?raw=true)  

위 두 가지 종류의 중요도 샘플링을 이해하기 위해, 하나의 반환 결과를 관찰한 후에 대해 first-visit 방법의 추정치를 살펴보자. 가중 중요도 샘플링의 기댓값은 $v_b(s)$이며 $v_\pi(s)$가 아니다. 통계적인 의미에서는 이는 편향된 추정치이다. 반면, 일반 중요도 샘플링에서는 항상 $v_\pi(s)$의 기대값이므로 편향되지 않는다. 하지만 이 추정치는 극단적일 수 있는데 비율이 10인 경우, 이는 관측된 경로가 대상 정책보다 행동 정책에서 10배 더 가능성이 높다는 것을 나타낸다.  
  
위와 같은 특징 때문에 두 종류의 중요도 샘플링의 first-visit 방법의 차이는 편향과 분산으로 표현 가능하다. 일반 중요도 샘플링은 편향되지 않지만 가중 중요도 샘플링은 편향되지만 점진적으로 0으로 수렴한다. 반면에 일반 중요도 샘플링의 분산은 일반적으로 무한대일 수 있다. 왜냐하면 비율의 분산은 무한대일 수 있기 때문이다. 반면에 가중 중요도 추정기의 경우 최대 가중치는 1이다. 실제로, 유한한 반환 값이라고 가정하면, 가중 중요도 샘플링 추정기의 분산은 비율의 분산이 무한대일지라도 0으로 수렴한다.  

every-visit 방법에선 일반 중요도 샘플링과 가중 중요도 샘플링의 모든 방문 방식은 모두 편향되지만, 샘플의 수가 증가함에 따라 편향은 점진적으로 0으로 수렴한다. 실제로는 모든 방문 방식이 선호되는 경우가 많다. 왜냐하면 방문한 상태를 추적할 필요가 없으며 근사화에 쉽게 확장할 수 있기 때문이다.  

