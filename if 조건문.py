# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # if 조건문

x = 10
if x == 10:
    print('10입니다.')

# +
x = int(input())          # 입력받은 값을 변수에 저장
 
if x == 10:               # x가 10이면
    print('10입니다.')    # '10입니다.'를 출력
 
if x == 20:               # x가 20이면
    print('20입니다.')    # '20입니다.'를 출력
# -

# # elif 

# +
x = 30
 
if x == 10:            
    print('10입니다.')
elif x == 20:           
    print('20입니다.')
else:                  # 조건식에 모두 만족하지 않을 때
    print('10도 20도 아닙니다.')
# -


