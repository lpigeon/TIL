# 2Layer Protocol 

## 2계층의 기능
2계층은 하나의 네트워크 대역, 즉, 같은 네트워크 상에 존재하는 여러 장비들 중에서 어떤 장비가 어떤 장비에게 보내는 데이터를 전달한다. <br>
추가적으로 <strong>오류제어, 흐름제어</strong>를 수행한다. <br>

## 2계층의 네트워크 크기
2계층은 하나의 네트워크 대역 LAN에서만 통신할 때 사용한다. <br>
다른 네트워크와 통신할 때는 항상 3계층이 도와주어야 한다. 다시 말해, LAN이 아니라 외부 네트워크와 통신할 때 2계층 만으로는 불가능 하다.<br>
3계층의 주소와 3계층의 프로토콜을 이용하여야만 다른 네트워크와 통신이 가능하다.<br>

## MAC 주소
LAN에서 통신할 때 사용하는 주소이다. 12개의 16진수로 이루어져 있으며 앞에 6개는 OUI 뒤에 6개는 고유 번호로 이루어져있다.<br>
* OUI : IEEE에서 부여하는 일종의 제조회사 식별 ID
* 고유번호 : 제조사에서 부여한 고유번호

## Ethernet 구조
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Ethernet_Type_II_Frame_format.svg/1920px-Ethernet_Type_II_Frame_format.svg.png" height="300px" width="100%"></p>

그림에서 볼 수 있듯이, 첫번째 6바이트(16진수 두개당 1byte)에는 목적지 주소(destination address) 이고 두번째 6바이트에는 보내는 주소(source address)이고 마지막 2바이트는 다음 상위 프로토콜이 무엇인지 알려주는 역할을 한다.<br>
TIP) Ethernet Type이 0x0800 이면 다음 프로토콜은 3계층 IPv4이고 0x0806이면 다음 프로토콜은 3계층 ARP이다.






