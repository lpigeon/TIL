# Basic of Network

네트워크란 노드들이 데이터를 공유할 수 있게 하는 디지털 전기통신망의 하나이다.<br>
노드란 네트워크에 속한 컴퓨터 또는 통신장비를 뜻하며 이러한 노드들이 서로 연결되어 데이터를 교환하는 것을 네트워크라 한다.<br>

## Network 분류

### 크기에 따른 분류
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Data_Networks_classification_by_spatial_scope.svg/1024px-Data_Networks_classification_by_spatial_scope.svg.png" height="300px" width="300px">
</p>

1. LAN(Local Area Network)
2. WAN(Wide Area Network)
3. MAN(Metropolitan Area Network)
4. VLAN, CAN, PAN etc.

크게 중요한 LAN과 WAN 중심으로 정리한다.<br>

* #### LAN
LAN은 가까운 지역끼리만 연결한 네트워크이다.<br>

* #### LAN
WAN은 멀리 있는 지역을 한데 묶은 네트워크이다.<br>
(여러개의 LAN들을 다시 하나로 묶은 네트워크)<br>

### 연결 형태에 따른 분류
1. Star 형 : 중앙 장비에 모든 노드가 연결
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/StarNetwork.svg/1024px-StarNetwork.svg.png" height="300px" width="300px">
</p>

2. Mesh 형 : 여러 노드들이 서로 그물처럼 연결
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/NetworkTopology-Mesh.svg/1280px-NetworkTopology-Mesh.svg.png" height="300px" width="300px">
</p>

3. Tree 형 : 나무의 가치처럼 계층 구조로 연결
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/5/5d/TreeTopology.png" height="300px" width="300px">
</p>

4. 링형, 버스형, 혼합형 etc.

실제 네트워크는 위에 연결형태를 혼합해서 사용

## Network 통신방식

1. 유니 캐스트 : 특정 대상과만 통신
2. 멀티 캐스트 : 특정 다수와 통신
3. 브로드 캐스트 : 네트워크에 있는 모든 대상과 통신

## Protocol
프로토콜은 일종의 약속, 양식이다. <br>

네트워크에서 노드와 노드가 통신할 때 어떤 노드가 어느 노드에게 어떤 데이터를 어떻게 보내는지 작성하기 위한 양식. <br>

각 프로토콜마다 각자의 양식이 있다.<br>

### 예시 프로토콜
* 가까운 곳과 연결할 때 : Ethernet 프로토콜(MAC 주소)
* 멀리 있는 곳과 연결할 때 : ICMP, IPv4, ARP(IP 주소)
* 여러가지 프로그램으로 연락할 때 : TCP, UDP (Port 번호)

## 패킷
여러 프로토콜들로 캡슐화 된 데이터

