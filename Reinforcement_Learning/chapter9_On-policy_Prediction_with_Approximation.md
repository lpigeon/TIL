# Chapter 9: On-policy Prediction with Approximation  

9장에서는 정책 $\pi$를 사용하여 생성된 경험으로부터 상태 가치 함수를 추정하기 위한 함수 근사를 다룬다.  
함수 근사를 위한 근사 값 함수는 테이블(기존의 방식)이 아닌 가중치 벡터 $w$로 표현된다.  
가중치 백터 $w$가 주어진 경우 상태 s의 근사 값 $\hat{v}(s,w)$ 는 $\hat{v}(s)$와 같다.  
이러한 함수 근사를 사용할 때 장점은 관측 불가능한 문제에도 적용할 수 있다는 거지만 단점으로 수렴가능성을 보장하지 못한다는 것이다.  

## Value-function Approximation  

DP, MC, TD는 각각의 업데이트 방식이 있었다. 이러한 업데이트를 $s$ -> $u$로 표현 할 수 있으며 이것으 상태 $s$에서 추정된 값이 업데이트 대상인 $u$와 비슷해야 한다는 의미로 해석 가능하다.  
지금까지의 모델들은 업데이트 방식이 단순해던 반면 함수 근사에서는 복잡한 방법을 사용한다. 값 예측을 위해 $s$ -> $u$를 훈련예제로 전달하여 근사 함수를 생성하여 이 함수를 바탕으로 값을 추정핟다.  

이러한 업데이트를 일반적인 훈련 예제로 사용한다면 인공 신경망, 의사 결정 트리등 다양한 종류의 다변량 회귀를 사용할 수 있다. 하지만 강화학습은 에이전트가 환경과 상호 작용 하는 온라인학습을 통해 학습하므로 점진적으로 획득한 데이터로 학습할 수 있는 방법이 필요하다. 또한, 강화 학습은 일반적으로 비정상적인 대상 함수(시간에 따라 변하는 대상 함수)를 처리 할 수 있는 근사 방법을 요구한다.  

## 9.2 The Prediction Objective ($\overline{VE}$)  

함수 근사에서는 테이블 형태와는 다르게 모든 상태의 값을 정확하게 맞추는 것은 불가능하다. 또한, 한 상태에서의 업데이트는 다른 상태에 영향을 미친다. 이러한 현상은 한 상태의 추정치를 정확하게 하면 다른 상태의 추정치는 덜 정확해지는 효과가 나오게 한다. 따라서 어떤 상태를 가장 중요하게 생각하는지 명시하는것이 필요한데 다음과 같은 상태 분포를 통해 각 상태 s가 오차에 얼마나 중요한지 나타낼수 있다.  

![state_dist](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/state_dist.png?raw=true)  

이때, 오차란 근사 값 $\hat{v}(s,w)$와 실제 값 $v_\pi(s)$ 사이의 차이의 제곱을 의미하며, 위 상태 분포를 적용하여 목적함수인 평균 제곱 값 오차(mean square value error)를 얻을 수 있다. 이것을 $\overline{VE}$라고 표현하며 식은 다음과 같다.

![ve](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/ve.png?raw=true)  

$\mu(s)$는 일반적으로 상태 s에서 소요되는 시간의 비율로 선택된다. on-policy 훈련에서는 이를 on-policy 분포(on-policy distribution)라고 한다.  

$\overline{VE}$ 관점에서 이상적인 목표는 전역 최적점(global optimum) 가중치 백터 $w^*$를 찾는 것이다. 선형 함수 근사기와 같은 단순 근사기에 경우 전역 최적점을 찾을 수 있지만, 인공 신경망이나 의사 결정 트리와 같은 복잡한 함수 근사기에서는 전역 최적점을 찾기에 힘들 수 있다. 이러한 경우 복잡한 함수 근사기는 지역 최적점(local optimum)으로 수렴할 수 있다. 하지만 이러한 수렴은 보장되지 않으며 최악의 경우 발산 할 가능성도 있다.  

$\overline{VE}$를 최적화 하기 위한 방법중 하나인 기울기 원칙을 기반으로 한 함수 근사방법과 선형 기울기 하강방법을 다음 쳅터에서 설명한다.  

## Stochastic-gradient and Semi-gradient Methods  

이번 챕터에서는 함수 근사를 위한 학습 방법 중 하나인 확률적 경사 하강법(SGD)에 대해 설명한다. SGD 방법은 함수 근사에서 가장 널리 쓰이는 방법중 하나이며 온라인 강화학습에 적합하다.  
경사 하강법에서 가중치 벡터$w$는 고정된 개수의 실수 값을 가진 열벡터로 표현된다. 또한, 모든 S에 대해 근사값 함수 $\hat{v}(s,w)$는 $w$에 대해 미분 가능한 함수이다. 가중치 벡터는 각 이산적인 시간(t = 0, 1, 2..)에 대해 업데이트해야 하므로 $w_t$로 표현한다.  

SGD에서는 매 step마다 다음과 같은 방식으로 $\overline{VE}$를 줄이는 방향으로 $w$를 업데이트한다.  

![sgd](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/sgd.png?raw=true)  

위 식에서 $\alpha$는 양의 step 크기 매개변수이고, ∇(Gradient)는 다음과 같은 편미분 벡터를 말한다.  

![gradient](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/gradient.png?raw=true)  

Gradient를 사용하기 때문에 경사 하강법(gradient descent)이라고 불리며, 매 업데이트마다 하나의 샘플을 사용해서 업데이트 하기 때문에 stochastic이라고 불린다. 확률적으로 하나의 샘플로써 업데이트하기 때문에 오차가 커지기도 작아지기도 한다. 따라서, $\alpha$가 다음과 같은 조건을 가져야 지역 최적점 수렴을 보장할 수 있다.  

![alpha_condition](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/alpha_condition.png?raw=true)  

만약, 타겟이 실제 가치값인 $v_\pi(S_t)$가 아니라 무작위 값이나 bootstrap한 값 등의 근사값 $U_t$일 경우, 다음과 같이 $U_t$값을 적용하여 근사할 수 있다.  

![sgd_ut](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/sgd_ut.png?raw=true)  

위 식이 지역 최적점 수렴을 보장할 수 있게 하기 위해선, $U_t$값이 비편향 추정치이고 $\alpha$가 앞서 말한 조건을 충족해야 한다.  

예를 들어 $U_t$ = $G_t$인 MC방법에 SGD를 적용할 때에 의사 코드는 다음과 같다. MC의 $G_t$는 $v_\pi(S_t)$의 비편향 추정치로 정의되어 지역 최적점 수렴을 보장한다.  

![gradient_mc](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/gradient_mc.png?raw=true)  

만약, bootstrap한 추정치의 경우, 타겟 자체가 현재 $w$값에 영향을 받아 편향(bias)이 생기게 된다. 그럴 경우 위 SGD식에서 타겟이 $w_t$에 대해 독립이 아니기 때문에 실제 Gradient가 아니게 된다. 이러 경우를 semi-gradient 방법이라고 한다.  
semi-gradient 방법은 일반적인 gradient 방법보다 안정적으로 수렴하지 않지만, 선형 함수의 경우를 비롯해 몇몇 주요한 경우에 신뢰성 있게 수렴을 한다. 또한, 학습속도를 크게 향상시키며 에피소드의 끝을 기다리지 않고 계속적이고 온라인으로 학습할 수 있다.  

아래는 전형적인 semi-gradient 방법인 semi-gradient TD(0)의 의사 코드이다. semi-gradient TD(0)의 $U_t$는 $R_(t+1)$ + $\gamma$ $v(S_{t+1},w)$이다.  

![semi_gradient_td](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/semi_gradient_td.png?raw=true)  

State aggregation은 함수 근사를 일반화하는 단순한 형태인데, 여러 상태들을 그룹으로 묶어 그룹별로 벡터 $w$의 값 하나를 배정하는 것이다. 그라디언트 벡터값에서 현재 업데이트되고 있는 상태 그룹의 요소만 1이고 나머지는 0이 된다.  

## Linear Methods  

함수 근사의 가장 중요한 특수한 경우 중 하나는 선형 함수이다. 선형 합수는 아래와 같이 나타낼수 있으며 벡터 $x(s)$는 상태를 표현한느 특징 벡터(feature vector)이고, 가중치 벡터 $w$와 선형으로 곱해져서 가지함수 $v$가 계산 된다. 이러한 선형 함수인 경우 그라디언트는 단순히 $x(s)$가 된다.  

![linear_method](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/linear_method.png?raw=true)  

![linear_method_gradient](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/linear_method_gradient.png?raw=true)  

이러한 선형 함수에 일반적인 SGD 업데이트를 사용하면 다음과 같이 업데이트를 할 수 있다.  

![sgd_linear](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/sgd_linear.png?raw=true)  

선형 SGD의 경우 하나의 최적점, 즉 전역 최저점 또는 그 근처에 수렴한다는 것을 보장한다. 예를 들어, Gradient MC 방법은 선형 함수 근사에서 $\overline{VE}$의 전역 최적점 수렴을 보장한다.  

반면, Semi-gradient TD(0)의 경우 선형 함수를 쓰면 수렴은 하지만 전역 최적점이 아닌 지역 최적점 근처로 수렴된다. 매 step t에서의 업데이트를 풀어쓰면 아래가 된다. 이때, $x_t$는 $x(S_t)$를 의미한다.  

![semi_gradient_td_linear](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/semi_gradient_td_linear.png?raw=true)  

만약 시스템이 안정된 상태(steady state)에 도달하게 되면, 어떤 주어진 wt에 대해 wt+1의 기대값은 아래와 같이 표기할 수 있다.  

![steady_state_system](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/steady_state_system.png?raw=true)  

만약 이 시스템이 수렴한다면 $w_{TD}$로 표시되는 가중치 벡터는 다음의 값으로 수렴됨을 알 수 있다.  

![w_td](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/w_td.png?raw=true)  

이러한 $w_{TD}$를 $TD$ $fixed$ $point$라고 부르며 선형함수를 사용하는 linear semi-gradient TD(0)는 이 포인트로 수렴한다.  
$TD$ $fixed$ $point$에서도 $\overline{VE}$가 가능한 최소 오차의 제한된 확장 범위 내에 있다는 것이 증명되었다.  

![min_w_td](https://github.com/lpigeon/TIL/blob/main/Reinforcement_Learning/img/chapter9/min_w_td.png?raw=true)  

위 식을 살펴보면 Gradient MC 방법에서 얻는 최소 오차보다 최대 1/1-$\gamma$ 배 이하이다. $\gamma$가 1에 가까울 수록 TD 방법의 점근적인 성능 손실이 상당할 것이다. 하지만 앞선 장에서 살펴 보았듯이 TD방법이 MC 방법에 비해 분산이 크게 감소하고 빠르다는것을 기억해야한다. 즉, 어떤 방법이 좋을 지는 문제에 따라 다른다.  

위의 bound와 비슷한 조건들이 다른 알고리즘들에도 적용이 되는데, 여기서 중요한 것은 on-policy distribution을 따라 상태들이 업데이트 되는 경우에 그렇다는 것이다. 다른 distribution으로 업데이트 된다면, bootstrapping 방법의 경우 수렴하지 않고 무한대로 발산할 수 있다.  

## Nonlinear Function Approximation: Artificial Neural Networks  

## Least-Squares TD  

