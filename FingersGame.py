import cv2
import mediapipe as mp
from keras import models
import tensorflow as tf # TensorFlow
import numpy as np
import os
import asyncio
from PIL import Image, ImageDraw, ImageFont
import random


# Грузим модель на старте
checkpoint_path = "/Users/lebedev/PycharmProjects/ObjectDetection/fingers.weights.h5"
model_path = '/Users/lebedev/PycharmProjects/ObjectDetection/fingers.2.1.model-tf2.11.h5'

# Проверьте, что файл модели существует
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Загружаем модель
model = models.load_model(model_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

waitingChar =[]
predictedChar =[]

sure = 0.90 # Уровень уверенности сети для рисования буквы (1 max)

alphabet = {0: "А", 1: "Б", 2: "В", 3: "Г", 4: "Д", 5: "Е", 6: "Ж", 7: "З",
            8: "И", 9: "К", 10: "Л", 11: "М", 12: "Н", 13: "О", 14: "П", 15: "Р",
            16: "С", 17: "Т", 18: "У", 19: "Ф", 20: "Х", 21: "Ц", 22: "Ч", 23: "Ш",
            24: "Ъ", 25: "Ы", 26: "Ь", 27: "Э", 28: "Ю", 29: "Я", 30: "spoke", 31: "Oops",}

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

async def draw_character(image, character, coords):
    try:
        # Логика рисования символа на frame

       # Переводим изображение в формат PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # Укажите путь к шрифту
        font = ImageFont.truetype("/Users/lebedev/PycharmProjects/ObjectDetection/a_AntiqueTrady.ttf", 70)  # Замени на свой путь и размер шрифта
        font2 = ImageFont.truetype("/Users/lebedev/PycharmProjects/ObjectDetection/a_AntiqueTrady.ttf", 20)  # Замени на свой путь и размер шрифта

        # Получаем размеры текста
        text_size = draw.textsize(character, font=font)
        text_x = int(coords[0] - text_size[0] / 2)
        text_y = int(coords[1] + text_size[1] / 2)


        ### обрабатывем ожидаемую букву

        print("waitingChar: ", str(waitingChar[0]))
        print("character: ", character)
        if str(waitingChar[0]) == character:
            waitingChar.pop(0)
            print ("Совпадение waitingChar == character")

            font3 = ImageFont.truetype("/Users/lebedev/PycharmProjects/ObjectDetection/a_AntiqueTrady.ttf",25)  # Замени на свой путь и размер шрифта
            draw.text((text_x, text_y), character, font=font3, fill=(255, 0, 0, 0))
            #pass

        # Рисуем текст
        draw.text((text_x, text_y), character, font=font, fill=(255, 255, 255, 0))
        draw.text((10, 10), "скажи: " + str(waitingChar[0]), font=font2, fill=(255, 0, 0, 0))

        # Возвращаем изображение в формат OpenCV

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred in draw_character: {e}")

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


async def process_hand_coordinates(hand_coordinates):
    # Здесь можете добавить код для обработки данных с помощью нейронной сети
    # Например, отправка данных на сервер или выполнение вычислений
    # print("Processing coordinates:", hand_coordinates)
    # print("Len:", len(hand_coordinates))

    #############
    hand_coordinates = np.asarray(hand_coordinates)
    hand_coordinates = hand_coordinates.reshape(1, -1)
    hand_coordinates = tf.convert_to_tensor(hand_coordinates, dtype=tf.float32)
    prediction = model.predict(hand_coordinates)
    predictedClass = np.argmax(prediction)
    AiSure = prediction.max()

    #await asyncio.sleep(0)  # синхронно спим =)
    if sure < AiSure:
        #print("Predicted:", alphabet[predictedClass], predictedClass)
        #print ("AiSure > limit:", AiSure)
        predictedChar.append(alphabet[predictedClass])

async def read_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удается захватить видео.")
            break

        # заполняем waitingChar буквой
        #waitingChar = 0
        if len(waitingChar) == 0:
            waitingChar.append(alphabet[random.randrange(30)])
            print("назначен waitingChar == ", waitingChar)

        #print("len(waitingChar):", len(waitingChar))
        #print("waitingChar:", waitingChar)




        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # # Проверка наличия обнаруженных рук
        # if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Получаем координаты средней фаланги среднего пальца (MCP)
                mid_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                h, w, _ = frame.shape
                coords = (int(mid_finger_mcp.x * w), int(mid_finger_mcp.y * h))

                # Рисуем точки для рук
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
####################
                # Отображение русской буквы по координатам
                if len(predictedChar) > 0:
                   #print ("predictedChar > 0")

                   try:
                       if predictedChar:
                           frame = await draw_character(frame, predictedChar.pop(0), coords)
                       else:
                           print("predictedChar is empty")
                   except Exception as e:
                       print(f"An error occurred: {e}")


        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[0]
            coordList = []  # список координат (заносим по 3: x, y, z)
            for id, lms in enumerate(my_hand.landmark):
                cx, cy, cz = lms.x, lms.y, lms.z
                # print("x:",cx, "y:",cy, "z:", cz)
                ## заносим в лист 3 кооридинаты
                coordList.append(cx)
                coordList.append(cy)
                coordList.append(cz)
            await process_hand_coordinates(coordList)


        cv2.imshow('Fingers', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def main():
    await read_video()

# Запуск асинхронного приложения
if __name__ == '__main__':
    asyncio.run(main())
