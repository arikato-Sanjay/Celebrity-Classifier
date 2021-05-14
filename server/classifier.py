import joblib
import json
import numpy as np
import base64
import pywt
from tempfile import NamedTemporaryFile
import streamlit as st
import cv2

__class_number_to_name = {}
__class_name_to_number = {}
__model = None


def classify_image(base64_data, file_path=None):
    image = get_cropped_img(file_path, base64_data)
    result = []
    for img in image:
        scale_img = cv2.resize(img, (32, 32))
        img_wav = wavelet_transform(img, 'db1', 5)
        scale_img_wav = cv2.resize(img_wav, (32, 32))
        stacked_img = np.vstack((scale_img.reshape(32 * 32 * 3, 1), scale_img_wav.reshape(32 * 32, 1)))
        length = 32 * 32 * 3 + 32 * 32
        final = stacked_img.reshape(1, length).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_prob': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dict': __class_name_to_number
        })
    return result


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def get_img_from_b64(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_img(img_path, b64str):
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')

    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_img_from_b64(b64str)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


st.cache(allow_output_mutation=True)


def load_artifacts():
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open('celebrity_dict.json', 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        with open('celebrity_model.pkl', 'rb') as f:
            __model = joblib.load(f)


def wavelet_transform(img, mode='haar', level=1):
    imgArray = img
    # converting to gray scale
    imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)
    # converting to float
    imgArray = np.float32(imgArray)
    # computing coeff
    coeffs = pywt.wavedec2(imgArray, mode, level=level)

    # processing coeffs
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imgArray_H = pywt.waverec2(coeffs_H, mode)
    imgArray_H *= 255
    imgArray_H = np.uint8(imgArray_H)

    return imgArray_H


def get_images(name, path, caption):
    name.image(path, caption=caption, width=130)


st.cache(allow_output_mutation=True)


def gui():
    global x
    error = '## Cannot classify image. Classifier was unable to detect face and two eyes'
    st.title('Sports Celebrity Classifier')

    img1, img2, img3, img4, img5 = st.beta_columns([1, 1, 1, 1, 1])
    get_images(img1, './virat.jpg', 'Virat Kohli')
    get_images(img2, './marry.jpg', 'Mary Kom')
    get_images(img3, './messi.jpg', 'Lionel Messi')
    get_images(img4, './saina.jpg', 'Saina Nehwal')
    get_images(img5, './sunil.jpg', 'Sunil Chhetri')

    file = st.file_uploader('Upload your image here (Choose from the above mentioned player)')
    temp_file = NamedTemporaryFile(delete=False)

    if file is not None:
        if st.button('Classify'):
            img, table = st.beta_columns([1, 1])
            img.image(file, width=300)
            temp_file.write(file.getvalue())
            output = classify_image(None, temp_file.name)
            if not output:
                st.error(error)
            else:
                for x in output:
                    key = x['class_prob']
                    data = {
                        'Mary Kom': [key[0]],
                        'Lionel Messi': [key[1]],
                        'Saina Nehwal': [key[2]],
                        'Sunil Chhetri': [key[3]],
                        'Virat Kohli': [key[4]]
                    }
                    st.write('## Probability Score')
                    st.dataframe(data)
                st.success('Celebrity is classified as: {}'.format(x['class']))


if __name__ == '__main__':
    load_artifacts()
    gui()
