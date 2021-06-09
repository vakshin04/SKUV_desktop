import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
mixer.init()
sound = mixer.Sound('sound.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml') 
#установка классификатора для нахождения лица
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml') 
#установка классификатора для нахождения левого глаза
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml') 
#установка классификатора для нахождения правого глаза

lbl = ['Closed', 'Open']

model = load_model('models/cnnmodel.h5') #модель для определения состояния глаз
path = os.getcwd()
cap = cv2.VideoCapture(0)#доступ к камере через OpenCV
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
#бесконечный цикл для захвата каждого кадра
while True:
    ret, frame = cap.read()#объект захвата, считываем и сохраняем каждый кадр
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #переводим в оттенки серого, т.к. OpenCV принимает только серые изображения
    
    #выполнение обнаружения лица
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)) 
    #возвращает массив обнаружений с координатами x,y и высотой, шириной границы объекта
    left_eye = leye.detectMultiScale(gray) #выполнение обнаружения левого глаза
    right_eye = reye.detectMultiScale(gray) #выполнение обнаружения правого глаза
    
    #черная подложка для состояния глаз и шкалы опасности
    cv2.rectangle(frame, (0, height - 50), (220, height), (0, 0, 0), thickness=cv2.FILLED)
    #перебираем найденный массив и рисуем граничные рамки для каждого объекта
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (105, 105, 105), 2)#выделение лица рамкой

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        # извлекаем данные правого глаза из изображения, выделяем границами прямоугольник глаз
        # выделяем границами прямоугольник глаз и берем нужный кусок
        # тут данные изображения глаза, которые мы отправляем в классификатор для исследования
        count = count + 1
        #конвертируем цветное изображение в оттенки серого
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        #изменяем размер до 24х24 пикселя, т.к. модель обучена на таких изображениях
        r_eye = cv2.resize(r_eye, (24, 24))
        #нормализуем данные, все значения между 0 и 1
        r_eye = r_eye / 255
        #развернем данные для внесения в классификатор
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        #анализируем
        rpred = model.predict_classes(r_eye)
        if rpred[0] == 1:#открыты
            lbl = 'Open'
        if rpred[0] == 0:#закрыты
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        #извлекаем данные левого глаза из изображения, выделяем границами прямоугольник глаз
        #выделяем границами прямоугольник глаз и берем нужный кусок
        #тут данные изображения глаза, которые мы отправляем с классификатор для исследования
        count = count + 1
        # конвертируем цветное изображение в оттенки серого
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        # изменяем размер до 24х24 пикселя, т.к. модель обучена на таких изображениях
        l_eye = cv2.resize(l_eye, (24, 24))
        # нормализуем данные, все значения между 0 и 1
        l_eye = l_eye / 255
        # развернем данные для внесения в классификатор
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        # анализируем
        lpred = model.predict_classes(l_eye)
        if lpred[0] == 1:#открыты
            lbl = 'Open'
        if lpred[0] == 0:#закрыты
            lbl = 'Closed'
        break
    #Управление счетчиком
    if rpred[0] == 0 and lpred[0] == 0:#увеличиваем, закрыты
        score = score + 2
        cv2.putText(frame, "Closed", (5, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:#уменьшаем, открыты
        score = score - 2
        cv2.putText(frame, "Open", (5, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #удерживаем крайние показатели в границах от 0 до 15
    if score < 0:
        score = 0
    if score > 15:
        score = 15
    # вывод показаний шкалы опасности
    cv2.putText(frame, 'Danger:' + str(score), (90, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)  
    if score == 15:
        #опасная ситуация
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)#сохраняем изображение и рисуем красную рамку
        try:
            sound.play()
        except:  # isplaying = False
            pass
        #тревожная красная рамка
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    #вписываем рамку в размер окна и отображаем на изображении
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'):#нажимаем 0 для закрытия системы
        break
# 4 кадра в секунду, скорость - 5 секунд для десктопной версии
cap.release()
cv2.destroyAllWindows()
