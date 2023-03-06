# 리스트 요소 추가

a = [1, 6, 8]
a.append(300)
a
len(a) # 4개

# 리스트 확장

a = [4, 6, 8]
a.extend(124, 4326)
a
len(a) # 5개

# 요소 넣기

a.insert(2, 111)
a # 인덱스 2에 111이 추가된다.
len(a)

# 요소 삭제
a = [1, 2, 3]
a.pop() # 마지막 요소 3 삭제
a

# 특정값 삭제
a.remove(2)
a


# 가장 작은 수, 가장 큰 수 구하기
a = [1, 2, 3, 4, 5]
smallest = a[0]
    for i in a:
    if i < smallest:
         smallest = i # 1 출력

a = [1, 2, 3, 4, 5]
largest = a[0]
    for i in a:
    if i > largest:
         largest = i # 5 출력

min(a) # 1 출력
max(a) # 5 출력

# 2차원 리스트

a = [[10, 20], [30, 40], [50, 60]]
a[0] [0] # 세로 인덱스 0, 가로 인덱스 0인 요소 출력
a[2][1] # 세로 인덱스 2, 가로 인덱스 0인 요소 출력
a[0][1] = 150 # 세로 인덱스 0, 가로 인덱스 1인 요소에 값 할당