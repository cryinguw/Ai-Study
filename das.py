# 반복분 사용 하기
from random import random

for i in range(10):  # hello world 10번 출력
    print('hello world!')


for i in range(10):  # 10번 출력 및 0-9
    print('hello world!', i)


for i in range(5, 12):    # 5부터 11까지 반복
    print('Hello, world!', i)


for i in range(0, 10, 2):    # 0부터 8까지 2씩 증가
    print('Hello, world!', i)


for i in range(10, 0, -1):    # 10에서 1까지 1씩 감소
     print('Hello, world!', i)


# range 대신 list를 넣으면 list의 요소를 꺼내면서 반복
a = [10, 20 ,30 ,40 ,50]
for i in a:
    print(i)


# 튜플도 가능
b = ('a', 'b', 'c')
for i in b:
    print(i)


# 각 글자를 공백으로 띄워서 출력
for letter in 'Python':
    print(letter, end=' ')


# 글자를 뒤집어서 출력
for letter in reversed('Python'):
     print(letter, end=' ')


# while 반복문 사용
i = 0                     # 초기식
while i < 100:            # while 조건식 (10미만 까지)
     print('Hello, world!')    # 반복할 코드
     i += 1                    # 변화식


# 입력 횟수대로 반복
count = int(input('반복할 횟수: '))

i = 0
while i < count:  # i가 count보다 작을 때 반복
    print('Hello, world!', i)
    i += 1


# 난수
import random

i = 0
while i != 3:  # 3이 아닐 때 계속 반복
    i = random.randint(1, 6)  # 1과 6 사이의 난수를 생성
    print(i)


# 반복문 멈추기
i = 0
while True:    # 무한 루프
    print(i)
    i += 1          # i를 1씩 증가시킴
    if i == 10:    # i가 10일 때
        break       # 반복문을 끝냄. while의 제어흐름을 벗어남


# continue
for i in range(10):       # 0부터 9까지 증가하면서 10번 반복
    if i % 2 == 1:         # i를 2로 나누었을 때 나머지가 1이면 홀수
        continue           # 아래 코드를 실행하지 않고 건너뜀
    print(i)


