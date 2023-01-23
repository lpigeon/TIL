# DFS
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/7/7f/Depth-First-Search.gif" height="300px" width="300px">
</p>
깊이 우선 탐색(Depth First Search) 또는 DFS라고 불리우는 탐색 알고리즘은 아래 설명할 BFS와 더불어 반드시 알아야 하는 탐색 알고리즘이다.

이름에서 알 수 있듯이 깊이를 우선시 하여 탐색한다. 한 루트가 정해지면 그 루트로 갈 수 있는한 깊숙이 들어가 탐색하고 만약 원하는 값이 없다면 다시 돌아가 다른 루트를 탐색한다. 여기서 다시 돌아간다는 것은 갈림길이 있던 장소까지 간다는 것이다.

이때, 다시 돌아간다는 것은 부모노드로 되돌아 간다는 의미이며 이 과정을 백트래킹(backtracking)이라고 한다.

단순 검색 속도 자체는 BFS보다 느리다.

## 장점
* 단지 현 경로상의 노드들만을 기억하면 되므로 저장공간의 수요가 비교적 적다.
* 목표노드가 깊은 단계에 있을 경우 해를 빨리 구할 수 있다.
## 단점
* 해가 없는 경로에 깊이 빠질 가능성이 있다. 따라서 실제의 경우 미리 지정한 임의의 깊이까지만 탐색하고 목표노드를 발견하지 못하면 다음의 경로를 따라 탐색하는 방법이 유용할 수 있다.
* 얻어진 해가 최단 경로가 된다는 보장이 없다. 이는 목표에 이르는 경로가 다수인 문제에 대해 깊이우선 탐색은 해에 다다르면 탐색을 끝내버리므로, 이때 얻어진 해는 최적이 아닐 수 있다는 의미이다. 

## Code
Python
```python
def dfs(graph, v, visited):
    # 현재 노드를 방문 처리
    visited[v] = True
    print(v, end=' ')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)
```
C++
```cpp
// DFS 함수 정의
void dfs(int x) {
    // 현재 노드를 방문 처리
    visited[x] = true;
    cout << x << ' ';
    // 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for (int i = 0; i < graph[x].size(); i++) {
        int y = graph[x][i];
        if (!visited[y]) dfs(y);
    }
}
```

# BFS
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Animated_BFS.gif" height="300px" width="300px">
</p>

너비 우선 탐색(Breadth-first search) 또는 BFS라고도 부른다. DFS와는 다르게 깊이가 아닌 넓이를 우선시 하여 탐색한다. 첫 시작 노드에서 연결된 노드를 다음 타겟으로 정하면서 진행된다.

## 장점
* 출발노드에서 목표노드까지의 최단 길이 경로를 보장한다.
## 단점
* 경로가 매우 길 경우에는 탐색 가지가 급격히 증가함에 따라 보다 많은 기억 공간을 필요로 하게 된다.
* 해가 존재하지 않는다면 유한 그래프(finite graph)의 경우에는 모든 그래프를 탐색한 후에 실패로 끝난다.
* 무한 그래프(infinite graph)의 경우에는 결코 해를 찾지도 못하고, 끝내지도 못한다.

## Code
Python
```python
from collections import deque
# BFS 함수 정의
def bfs(graph, v, visited):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([v])
    # 현재 노드를 방문 처리
    visited[v] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end=' ')
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
```
C++
```cpp
// BFS 함수 정의
void bfs(int start) {
    queue<int> q;
    q.push(start);
    // 현재 노드를 방문 처리
    visited[start] = true;
    // 큐가 빌 때까지 반복
    while(!q.empty()) {
    	// 큐에서 하나의 원소를 뽑아 출력
        int x = q.front();
        q.pop();
        cout << x << ' ';
        // 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for(int i = 0; i < graph[x].size(); i++) {
            int y = graph[x][i];
            if(!visited[y]) {
                q.push(y);
                visited[y] = true;
            }
        }
    }
}
```


<br>

### 참고
* https://ko.wikipedia.org/wiki/%EB%84%88%EB%B9%84_%EC%9A%B0%EC%84%A0_%ED%83%90%EC%83%89
* https://ko.wikipedia.org/wiki/%EA%B9%8A%EC%9D%B4_%EC%9A%B0%EC%84%A0_%ED%83%90%EC%83%89
* https://github.com/ndb796/python-for-coding-test
