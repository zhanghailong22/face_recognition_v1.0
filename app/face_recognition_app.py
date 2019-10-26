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
from io import BytesIO
import psycopg2
import face_recognition
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import redis

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# 图像录入
@app.route('/v1/face_recognition/upload', methods=['POST'])
def upload():
    # 连接redis
    pool = redis.ConnectionPool(host='10.97.97.97', password="root", port=6379)
    # pool = redis.ConnectionPool(host='redis', port=6379)
    r = redis.Redis(connection_pool=pool)
    if 'file' not in request.files:
        return jsonify({'code': 500, 'msg': '没有文件'})
    files = request.files.getlist("file")
    num = 0
    for file in files:
        img = file.read()
        face_upload(r, img, file)
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
    result = face_compare(file1, file2)
    return jsonify(result)



# 人脸搜索
@app.route('/v1/face_recognition/search_images', methods=['POST'])
def search_images():
    # 连接redis
    pool = redis.ConnectionPool(host='10.97.97.97', password="root", port=6379)
    # pool = redis.ConnectionPool(host='redis', port=6379)
    r = redis.Redis(connection_pool=pool)
    if 'file' not in request.files:
        return jsonify({'code': 500, 'result': '没有文件'})
    file = request.files['file']
    find_faces = face_find(r, file)
    #    print(find_faces)
    return jsonify({'查询到的人脸个数': len(find_faces),'人脸信息': find_faces})

    # number, find_names = api.face_find(r, file)

    # return jsonify({'图片中人脸的个数': number, '查询到的人脸个数': len(find_names), \
    #                 "find_names": [str(name, 'utf-8') for name in find_names]})


# 视频监控
@app.route('/v1/face_recognition/search_video', methods=['POST'])
def search_video():
    # 连接redis
    pool = redis.ConnectionPool(host='10.97.97.97', password="root", port=6379)
    # pool = redis.ConnectionPool(host='redis', port=6379)
    r = redis.Redis(connection_pool=pool)
    face_names = video_find(r)
    return jsonify({'names': face_names})


# 刷新redis，将mysql数据库中的人脸图片生成特征向量并导入redis数据库
@app.route('/v1/face_recognition/refresh_redis', methods=['POST'])
def update_redis():
    # 连接redis
    pool = redis.ConnectionPool(host='10.97.97.97', password="root", port=6379)
    # pool = redis.ConnectionPool(host='redis', port=6379)
    r = redis.Redis(connection_pool=pool)
    number = refresh_redis(r)
    return jsonify({'redis中人脸数目': number, 'result': '刷新成功'})


# 允许上传的文件类型：png、jpg、jpeg。
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def face_upload(r, img, file):
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return jsonify({'code': 500, 'error': '人脸数量有误'})
    if file and allowed_file(file.filename):
        # 保存在mysql数据库中
        conn = psycopg2.connect(host="10.97.97.98", user="postgres", password="root", database="face_images", port=5432)
        cur = conn.cursor()
        cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('image_data',))
        if cur.fetchone()[0] == 0:
            command = "create table if not exists image_data (name text PRIMARY KEY NOT NULL, image bytea NOT NULL);"  # create table cataract
            cur.execute(command)
            r_command = "create rule  r_image_data as on insert to image_data where exists (select 1 from image_data where name =new.name) do instead nothing;"
            cur.execute(r_command)
            conn.commit()  # commit the changes
        name = file.filename[0:-4]
        command = "insert into image_data(name, image) values(%s, %s);"  # create table cataract
        params = (name, psycopg2.Binary(img))
        cur.execute(command, params)
        conn.commit()  # commit the changes
        # sql = 'insert ignore into image_data (name,image) values(%s, %s);'
        # data = [(name, pymysql.Binary(img))]
        # cursor.executemany(sql, data)
        # conn.commit()
        cur.close()
        conn.close()
    else:
        return jsonify({'error': '图片格式错误'})
    face_encodings = face_recognition.face_encodings(image, face_locations)
    # 连数据库
    # 录入人名-对应特征向量
    r.set(name, face_encodings[0].tobytes())


def face_find(r, file):
    if allowed_file(file.filename) == 0:
        # The image file seems valid! Detect faces and return the result.
        return jsonify({'error': '图片格式错误'})
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    # 取出所有的人名和它对应的特征向量
    names = r.keys()
    faces = r.mget(names)
    # 组成矩阵，计算相似度（欧式距离）
    find_names = []
    number = len(face_encodings)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([np.frombuffer(x) for x in faces], face_encoding)
        num = 0
        for name, match in zip(names, matches):
            num = num + 1
            if match:
                face_name = {'name': str(name, 'utf-8'), '人脸位置': {'top': top, 'right': right, 'bottom': bottom, 'left': left}}
                find_names.append(face_name)
                break
            if matches[-1] == 0 and len(matches) == num:
                face_name = {'name': 'unknown', '人脸位置': {'top': top, 'right': right, 'bottom': bottom, 'left': left}}
                find_names.append(face_name)

    return find_names


def face_compare(file1, file2):
    image1 = face_recognition.load_image_file(file1)
    face_locations1 = face_recognition.face_locations(image1)
    face_encodings1 = face_recognition.face_encodings(image1)
    image2 = face_recognition.load_image_file(file2)
    face_locations2 = face_recognition.face_locations(image2)
    if len(face_locations2) != 1:
        return jsonify({'code': 500, 'error': '人脸数量有误'})
    face_encodings2 = face_recognition.face_encodings(image2)[0]
    face_distances = face_recognition.face_distance(face_encodings1, face_encodings2)
    result = []
    for (top, right, bottom, left), distance in zip(face_locations1, face_distances):
        face_local = {'相似度':1-distance, '人脸位置': {'top': top, 'right': right, 'bottom': bottom, 'left': left}}
        result.append(face_local)
    return result


# 视频
def video_find(r):
    # 连数据库
    # 取出所有的人名和它对应的特征向量
    names = r.keys()
    faces = r.mget(names)
    # 组成矩阵，计算相似度（欧式距离）
    face_names = video_camera(names, faces)
    return face_names


def refresh_redis(r):
    # 1.连接mysql数据库
    conn = psycopg2.connect(host="10.97.97.98", user="postgres", password="root", database="face_images", port=5432)
    # 2.创建游标
    cursor = conn.cursor()
    sql = "select * from image_data"
    cursor.execute(sql)  # 执行sql
    # 查询所有数据，返回结果默认以元组形式，所以可以进行迭代处理
    for i in cursor.fetchall():
        name = i[0]  # get name
        data = i[1]  # get data
        #file = Image.open(BytesIO(data))
        image = face_recognition.load_image_file(BytesIO(data))
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # 录入人名-对应特征向量
        r.set(name, face_encodings[0].tobytes())
    cursor.close()
    conn.close()
    number = len(r.keys())
    return number


def video_camera(names, known_faces):
    video_capture = cv2.VideoCapture(0)
    # Initialize some variables
    face_locations = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces([np.frombuffer(x) for x in known_faces], face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance([np.frombuffer(x) for x in known_faces], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = str(names[best_match_index], 'utf-8')

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return face_names
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9999)
