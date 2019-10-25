#!/usr/bin/python
# -*- coding: utf-8 -*-
# This is an app application for face recognition microservices.
# Its functions are: face upload, face recognition, Video face recognition, etc.
# The restfull api is http://127.0.0.1:9999. Database is redis, used to store
# the feature vector of the face.
# Store face images in data volume mysql
# Author e-mail:zhanghailong22@huawei.com
#
# -*- coding: utf-8 -*-

from flask import Flask, Response, request, jsonify
import redis
import app.api as api

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 连接redis
pool = redis.ConnectionPool(host='10.97.97.97', password="root", port=6379)
# pool = redis.ConnectionPool(host='redis', port=6379)
r = redis.Redis(connection_pool=pool)


# 图像录入
@app.route('/v1/face_recognition/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'code': 500, 'msg': '没有文件'})
    files = request.files.getlist("file")
    num = 0
    for file in files:
        img = file.read()
        api.face_upload(r, img, file)
        num = num + 1
    return jsonify({'number': num, 'result': '录入成功'})


# 人脸对比
@app.route('/v1/face_recognition/compare_image', methods=['POST'])
def compare_image():
    if 'file' not in request.files:
        return jsonify({'code': 500, 'msg': '没有文件'})
    files = request.files.getlist("file")
    if len(files)!=2:
        return jsonify({'msg': '文件不足'})
    file1 = files[0]
    file2 = files[1]
    result = api.face_compare(file1, file2)
    print(result)
    return jsonify(result)



# 人脸搜索
@app.route('/v1/face_recognition/search_images', methods=['POST'])
def search_images():
    if 'file' not in request.files:
        return jsonify({'code': 500, 'result': '没有文件'})
    file = request.files['file']
    find_faces = api.face_find(r, file)
    #    print(find_faces)
    return jsonify({'查询到的人脸个数': len(find_faces),'人脸信息': find_faces})

    # number, find_names = api.face_find(r, file)

    # return jsonify({'图片中人脸的个数': number, '查询到的人脸个数': len(find_names), \
    #                 "find_names": [str(name, 'utf-8') for name in find_names]})


# 视频监控
@app.route('/v1/face_recognition/search_video', methods=['POST'])
def search_video():
    face_names = api.video_find(r)
    return jsonify({'names': face_names})


# 刷新redis，将mysql数据库中的人脸图片生成特征向量并导入redis数据库
@app.route('/v1/face_recognition/refresh_redis', methods=['POST'])
def update_redis():
    number = api.refresh_redis(r)
    return jsonify({'redis中人脸数目': number, 'result': '刷新成功'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9999)
