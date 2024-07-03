import cv2 as cv
import numpy as np
import argparse
import math
import pytesseract

# # Вычисляет координаты текстовых блоков и добавляет их в списки rects и confidences.
# # @param scores: Содержит вероятности наличия текста в каждой ячейке сетки (feature map).
# # @param geometry: Содержит координаты и размеры предсказанных текстовых блоков.
# # @param scoreThresh: Порог уверенности, ниже которого предсказания игнорируются.
# def decode(scores, geometry, scoreThresh):
#     # Определение размеров scores (numRows) и geometry (numCols)
#     numRows, numCols = scores.shape[2:4]
#
#     # Пустые списки для хранения координат прямоугольников и уверенностей
#     rects = []
#     confidences = []
#
#     # Цикл по строкам сетки
#     for y in range(numRows):
#         # Извлечение данных вероятностей текста для текущей строки
#         scoresData = scores[0, 0, y]
#
#         # Извлечение данных геометрии для текущей строки
#         xData0 = geometry[0, 0, y]
#         xData1 = geometry[0, 1, y]
#         xData2 = geometry[0, 2, y]
#         xData3 = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]
#
#         # Цикл по столбцам сетки
#         for x in range(numCols):
#             # Получение уверенности (score) для текущей ячейки
#             score = scoresData[x]
#
#             # Проверка, превышает ли уверенность порог scoreThresh
#             if score < scoreThresh:
#                 continue  # Пропустить текущую ячейку, если уверенность ниже порога
#
#             # Вычисление смещений относительно начала координат
#             offsetX = x * 4.0
#             offsetY = y * 4.0
#
#             # Извлечение угла поворота текстового блока
#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)
#
#             # Вычисление высоты и ширины текстового блока
#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]
#
#             # Вычисление координат прямоугольника
#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)
#
#             # Добавление координат и уверенности в соответствующие списки
#             rect = ((offsetX, offsetY), (w, h), -angle * 180.0 / np.pi)
#             rects.append(rect)
#             confidences.append(float(score))
#
#     # Возвращение списка координат прямоугольников и списка уверенностей
#     return rects, confidences
#
#
# if __name__ == "__main__":
#     # # Парсинг аргументов командной строки
#     # parser = argparse.ArgumentParser(description='Text detection using EAST model')
#     # parser.add_argument('image', help='Path to input image')
#     # args = parser.parse_args()
#
#     # Загрузка модели
#     net = cv2.dnn.readNet("frozen_east_text_detection.pb")
#
#     # # Путь к изображению
#     # image_path = args.image
#
#     # Путь к изображению
#     image_path = "C:\\Users\\Acer\\PycharmProjects\\tensorflow_project\\pass.jpg"
#
#     # # Загрузка изображения
#     # frame = cv2.imread(image_path)
#
#     # # Путь к изображению
#     # image_path = args.image
#
#     # Чтение изображения с помощью OpenCV
#     frame = cv2.imread(image_path)
#     # if frame is None:
#     #     print(f"Error: Could not open or find the image '{image_path}'")
#     #     exit()
#
#     # Размеры входного изображения (ПЕРЕДАВАТЬ ЧЕРЕЗ АРГУМЕНТЫ?)
#     inpWidth = 320
#     inpHeight = 320
#
#     # Создание блоба из изображения
#     # 1 параметр - само изображение
#     # 2 - масштабирование пикселей - обычно 1.0
#     # 3 - размеры изображения
#     # 4 -усреднение пикселей - не особо понял
#     # 5 - имеются 3 слоя(RGB) true -  меняет каналы местами R и B
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
#
#
#     #### Передаем входные данные через сеть и получаем выходные результаты
#     # Установим входные данные для сети
#     net.setInput(blob)
#
#     # Определим выходные слои
#     outputLayers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
#
#     # Выполним прямой проход через сеть
#     output = net.forward(outputLayers)
#
#     # Разделим выход на оценки и геометрию
#     scores = output[0]
#     geometry = output[1]
#
#     # Порог уверенности
#     confThreshold = 0.8
#
#     # Декодирование предсказаний
#     boxes, confidences = decode(scores, geometry, confThreshold)
#
#     # Порог для немаксимального подавления
#     nmsThreshold = 0.3
#
#     # Применение немаксимального подавления
#     indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
#
#     # # Рисование результатов на изображении
#     # for i in indices:
#     #     (startX, startY, endX, endY) = boxes[i[0]]
#     #     cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
#     # Прорисовка прямоугольников на изображении
#     for i in indices:
#         box = cv2.boxPoints(boxes[i])
#         box = np.int32(box)
#         cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
#
#
#     # Показать результирующее изображение
#     cv2.imshow("Text Detection", frame)
#
#     # Сохранение изображения с выделенными текстовыми областями
#     output_image_path = "output_detected_text.jpg"
#     cv2.imwrite(output_image_path, frame)
#     print(f"Результат сохранен в файл: {output_image_path}")
#
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

parser.add_argument('--device', default="cpu", help="Device to inference on")


args = parser.parse_args()


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    # Load network
    net = cv.dnn.readNet(model)
    if args.device == "cpu":
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    # Create a new named window
    kWinName = "Blur photos"
    # cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # Get scores and geometry
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

        # # Recognize text inside each bounding box
        # texts = recognize_text(frame, [boxes[i] for i in indices])

        for i in indices:
            vertices = cv.boxPoints(boxes[i])  # Get the rotated rectangle's vertices
            # Scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            # Draw lines between the vertices to form the rectangle

            # Вычисляем границы прямоугольника
            x1, y1 = vertices[1]
            x2, y2 = vertices[3]

            # Вырезаем область с текстом из кадра
            roi = frame[int(y1):int(y2), int(x1):int(x2)]

            # Применяем размытие с использованием гауссова фильтра
            blurred_roi = cv.GaussianBlur(roi, (45, 45), 0)

            # Заменяем область с текстом на размытую версию
            frame[int(y1):int(y2), int(x1):int(x2)] = blurred_roi


            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv.line(frame, p1, p2, (206, 212, 210), 1, cv.LINE_AA)


        # Put efficiency information
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


        # Display the frame
        # cv.imshow(kWinName, frame)
        cv.imwrite("output.jpg", frame)









