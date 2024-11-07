#
# Сканирование 21-й координаты (в осях x, y, z)любой руки с помощью OpenCV и mediapipe
# Файл позволяет собирать коордитаны и писать в файл csv для дальнейшего обучений нейронной сети.
# (c) LEbEdEV www.lawnow.ru
# датасет:
# 1 - преимущественно окончания жестов
# 2 - сквозные буквы с начала до конца жеста
# 3 - https://youtu.be/jtbwEalS0CE?si=eAT6HI1jGFFMGX04

import cv2
import mediapipe as mp
import pandas as pd
from datetime import date
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_detection_confidence=0.5)

#!!!!!
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# define a video capture object
vid = cv2.VideoCapture(0)

coordinats = []
coordList = [] # список координат (заносим по 3: x, y, z)
classLetter = 24

alphabet = {0: "А",  1: "Б",  2: "В",  3: "Г",  4: "Д",  5: "Е",  6: "Ж",  7: "З",
            8: "И",  9: "К",  10: "Л",  11: "М",  12: "Н",  13: "О",  14: "П",  15: "Р",
            16: "С",  17: "Т",  18: "У",  19: "Ф",  20: "Х",  21: "Ц",  22: "Ч",  23: "Ш",
            24: "Ъ", 25: "Ы", 26: "Ь", 27: "Э", 28: "Ю", 29: "Я", 30: "spoke", 31: "Fuck you",}



while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if ret:
        assert not isinstance(frame, type(None)), 'frame not found'

    # тестирум пришло ли что с камеры

    # if frame is not None:
    #     print('variable is not None')
    #     # print(frame.shape)
    #     pass
    # else:
    #     #print('variable is None')
    #     pass


    h, w, ret = frame.shape
    results = hands.process(frame)

    ## синие шарики, если хотите:

    # if result.multi_hand_landmarks:
    #         for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
    #                 #print(id, ":", lm)
    #
    #                 cx, cy = int(lm.x * w), int(lm.y * h)
    #                 #print (id,";", cx,":",cy)
    #                 cv2.circle(frame, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
    #         if id == 8:
    #                 cvw.circle(frame, (cx, cy), 25, (0, 255, 0))


    ## рисуем стиль по умолчанию - просто для красоты ;)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = hands.process(frame)

################

    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        coordList = []  # список координат (заносим по 3: x, y, z)
        for id, lms in enumerate(my_hand.landmark):
            # print("CURRENT ID", id)

            ## вариант для перехода в координаты на картинке (пиксели)
            #h, w, c = frame.shape
            #cx, cy = int(lms.x * w), int(lms.y * h)

            cx, cy, cz = lms.x, lms.y, lms.z
            #print("x:",cx, "y:",cy, "z:", cz)

            ## заносим в лист 3 кооридинаты
            coordList.append(cx)
            coordList.append(cy)
            coordList.append(cz)
            if id == 20:
                ## добавляем класс для нейронки
                coordList.append(classLetter)

                # проверка 21*3(x,y,z) + classLetter = 64
                #print("CURRENT ID:", id, "coordList:", coordList)

                # Вывод собранных координат
                #print(coordList)
                coordinats.append(coordList)

            if len(coordinats) > 1000: # Циклов до сохранения в файл и обнуления списка координат
                import datetime
                now = datetime.datetime.now()

                print(" ################## saving ##################")

                dfCoordinats = pd.DataFrame(coordinats) # создаем
                dfCoordinats.to_csv(str(classLetter)+"-"+str(now.year)+"-"+str(now.month)+"-"+str(now.hour) +"-"+str(now.minute)+"-"+ str(now.second)+".csv", index=False, header=False)
                coordinats = [] # обнуляем сохраненный список
                time.sleep(5) # тормозим для визуализации записи файла


### версия GPT№

    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         # Создаем пустой массив для хранения всех точек
    #         all_points = []
    #         # Перебираем все точки левой руки и добавляем их координаты в массив
    #         for point in hand_landmarks.landmark:
    #             all_points.append((point.x, point.y, point.z))
    #             #print(all_points)
### версия GPT

    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
