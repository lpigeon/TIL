# CSRF

CSRF는 교차 사이트 요청 위조(Cross Site Request Forgery)의 약어이며 임의 이용자의 권한으로 임의 주소에 HTTP 요청을 보낼 수 있는 취약점이다.<br>
CSRF 공격을 성공시키기 위해서는 공격자가 작성한 악성 스크립트를 이용자가 실행하여야 한다. CSRF 공격 스크립트는 HTML 또는 Javascript를 통해 작성할 수 있는데 img 또는 form태그를 사용하는 방법이 있다. <br>


## CSRF 스크립트 예시
다음은 dreamhack에서 제공한 CSRF 스크립트 예시이다.<br>

* HTML img 태그 공격 코드 예시
```HTML
    <img src='http://bank.dreamhack.io/sendmoney?to=dreamhack&amount=1337' width=0px height=0px>
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

* Javascript 공격 코드 예시
```javascript
    /* 새 창 띄우기 */
    window.open('http://bank.dreamhack.io/sendmoney?to=dreamhack&amount=1337');
    /* 현재 창 주소 옮기기 */
    location.href = 'http://bank.dreamhack.io/sendmoney?to=dreamhack&amount=1337';
    location.replace('http://bank.dreamhack.io/sendmoney?to=dreamhack&amount=1337');
```

## XSS와 CSRF의 차이
* 공통점 : 두 개의 취약점은 모두 클라이언트를 대상으로 하는 공격이며, 이용자가 악성 스크립트가 포함된 페이지에 접속하도록 유도해야 한다.
* 차이점 : 두 개의 취약점은 공격에 있어 서로 다른 목적을 가집니다. XSS는 인증 정보인 세션 및 쿠키 탈취를 목적으로 하는 공격이며, 공격할 사이트의 오리진에서 스크립트를 실행시킨다.
CSRF는 이용자가 임의 페이지에 HTTP 요청을 보내는 것을 목적으로 하는 공격이다. 또한, 공격자는 악성 스크립트가 포함된 페이지에 접근한 이용자의 권한으로 웹 서비스의 임의 기능을 실행할 수 있다.