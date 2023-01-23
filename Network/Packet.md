# 패킷(Packet)
패킷이란 네트워크 상에서 전달되는 데이터를 통칭하는 말로 네트워크에서 전달하는 데이터의 형식화된 블록이다.<br>
페킷은 제어 정보와 사용자 데이터로 이루어지며 사용자 데이터는 <strong>페이로드</strong>라고도 한다.<br>

## 패킷의 구성요소
패킷은 크게 헤더(Header), 페이로드(Payload), 풋터(Footer) 로 구성되어 있으며 대부분의 프로토콜은 풋터를 가지고 있지 않지만 Ethernet은 가지고 있다. - wireshark로 확인가능<br>

## 패킷을 이용한 통신과정
* 캡슐화(encapsulation)
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/UDP_encapsulation.svg/1920px-UDP_encapsulation.svg.png" height="300px" width="500px"></p>
<br>
헤더와 페이로드를 하나의 페이로드로 헤서 다른 헤더를 붙이고 또 그 데이터를 페이로드로 해서 다른 헤더를 붙이는 일련의 과정이다.<br>
특징은 헤더와 페이로드를 만들때의 계층의 순서인데, 예를 들어, Ethernet(2계층) - IPv4(3계층) - TCP(4계층) - 데이터 는 가능하지만 IPv4(3계층) - Ethernet(2계층) - TCP(4계층) - 데이터 는 불가능 하다.<br>
프로토콜에 따라 3계층 - 3계층 등 같은 계층을 붙이는 경우는 가능하다.<br><br>

* 디캡슐화(decapsulation)
캡슐화의 반대 과정이다. 하위계층부터 상위계층으로 디캡슐화 된다.

## PDU(Protocol Data Unit)
프로토콜 데이터 단위(Protocol Data Unit)는 데이터 통신에서 상위 계층이 전달한 데이터에 붙이는 제어정보를 뜻한다.<br>

* 1계층의 PDU – 비트(스트림)
* 2계층의 PDU – 전달정보(프레임)
* 3계층의 PDU – 패킷 혹은 UDP의 데이터그램
* 4계층의 PDU – TCP 세그먼트
* 5-6-7계층의 PDU PDU – 메시지, 데이터