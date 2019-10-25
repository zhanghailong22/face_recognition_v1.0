#!/usr/bin/python
# -*- coding: utf-8 -*-
# test face_recognition_app.py

import requests

# 录入
url = "http://127.0.0.1:9999/v1/face_recognition/upload"

filepath1 = 'D:\localtest/face_recognition_v3.0\examples/biden.jpg'
split_path1 = filepath1.split('/')
filename1 = split_path1[-1]

filepath2 = 'D:\localtest/face_recognition_v3.0\examples/obama.jpg'
split_path2 = filepath2.split('/')
filename2 = split_path2[-1]

file1 = open(filepath1, 'rb')
file2 = open(filepath2, 'rb')
files = [('file', (filename1, file1, 'image/jpg')), ('file', (filename2, file2, 'image/jpg'))]
print(files)
r = requests.post(url, files=files)
result = r.text
print(result)

# # 查询
url = "http://127.0.0.1:9999/v1/face_recognition/search_images"

filepath = 'D:\localtest/face_recognition_v3.0\examples/two_people.jpg'
split_path = filepath.split('/')
filename = split_path[-1]
print(filename)

file = open(filepath, 'rb')
files = {'file': (filename, file, 'image/jpg')}

r = requests.post(url, files=files)
result = r.text
print(result)

url = "http://127.0.0.1:9999/v1/face_recognition/compare_image"
filepath1 = 'D:\localtest/face_recognition_v3.0\examples/two_people.jpg'
split_path1 = filepath1.split('/')
filename1 = split_path1[-1]

filepath2 = 'D:\localtest/face_recognition_v3.0\examples/obama.jpg'
split_path2 = filepath2.split('/')
filename2 = split_path2[-1]

file1 = open(filepath1, 'rb')
file2 = open(filepath2, 'rb')
files = [('file', (filename1, file1, 'image/jpg')), ('file', (filename2, file2, 'image/jpg'))]
r = requests.post(url, files=files)
result = r.text
print(result)

# # 视频监控
# url = "http://127.0.0.1:9999/v1/face_recognition/search_video"
#
# r = requests.post(url)
# result = r.text
# print(result)
#
# 刷新redis
url = "http://127.0.0.1:9999/v1/face_recognition/refresh_redis"

r = requests.post(url)
result = r.text
print(result)
