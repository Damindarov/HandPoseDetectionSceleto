# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import mediapipe as mp
from tkinter import *
import time
import mouse as mouse
import pyrealsense2 as rs
import numpy as np
import cv2
import math
from pip._vendor.msgpack.fallback import xrange

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#class for drawing interface
class Paint(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.color = "black"
        self.brush_size = 10
        self.setUI()

    def quit_program(self):
        self.root.destroy()

    def set_color(self, new_color):
        self.color = new_color

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

    def setUI(self):
        self.parent.title("Pythonicway PyPaint")
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(6,
                             weight=1)
        self.rowconfigure(2, weight=1)

        self.canv = Canvas(self, bg="white")  # Создаем поле для рисования, устанавливаем белый фон
        self.canv.grid(row=2, column=0, columnspan=7,
                       padx=5, pady=5,
                       sticky=E + W + S + N)  # Прикрепляем канвас методом grid. Он будет находится в 3м ряду, первой колонке, и будет занимать 7 колонок, задаем отступы по X и Y в 5 пикселей, и заставляем растягиваться при растягивании всего окна
        self.canv.bind("<B1-Motion>",
                       self.draw)  # Привязываем обработчик к канвасу. <B1-Motion> означает "при движении зажатой левой кнопки мыши" вызывать функцию draw

        color_lab = Label(self, text="Color: ")  # Создаем метку для кнопок изменения цвета кисти
        color_lab.grid(row=0, column=0,
                       padx=6)  # Устанавливаем созданную метку в первый ряд и первую колонку, задаем горизонтальный отступ в 6 пикселей

        red_btn = Button(self, text="Red", width=10,
                         command=lambda: self.set_color(
                             "red"))  # Создание кнопки:  Установка текста кнопки, задание ширины кнопки (10 символов), функция вызываемая при нажатии кнопки.
        red_btn.grid(row=0, column=1)  # Устанавливаем кнопку

        # Создание остальных кнопок повторяет ту же логику, что и создание
        # кнопки установки красного цвета, отличаются лишь аргументы.

        green_btn = Button(self, text="Green", width=10,
                           command=lambda: self.set_color("green"))
        green_btn.grid(row=0, column=2)

        blue_btn = Button(self, text="Blue", width=10,
                          command=lambda: self.set_color("blue"))
        blue_btn.grid(row=0, column=3)

        black_btn = Button(self, text="Black", width=10,
                           command=lambda: self.set_color("black"))
        black_btn.grid(row=0, column=4)

        white_btn = Button(self, text="White", width=10,
                           command=lambda: self.set_color("white"))
        white_btn.grid(row=0, column=5)

        clear_btn = Button(self, text="Clear all", width=10,
                           command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=0, column=6, sticky=W)

        size_lab = Label(self, text="Brush size: ")
        size_lab.grid(row=1, column=0, padx=5)
        one_btn = Button(self, text="Two", width=10,
                         command=lambda: self.set_brush_size(2))
        one_btn.grid(row=1, column=1)

        two_btn = Button(self, text="Five", width=10,
                         command=lambda: self.set_brush_size(5))
        two_btn.grid(row=1, column=2)

        five_btn = Button(self, text="Seven", width=10,
                          command=lambda: self.set_brush_size(7))
        five_btn.grid(row=1, column=3)

        seven_btn = Button(self, text="Ten", width=10,
                           command=lambda: self.set_brush_size(10))
        seven_btn.grid(row=1, column=4)

        ten_btn = Button(self, text="Twenty", width=10,
                         command=lambda: self.set_brush_size(20))
        ten_btn.grid(row=1, column=5)

        twenty_btn = Button(self, text="Fifty", width=10,
                            command=lambda: self.set_brush_size(50))
        twenty_btn.grid(row=1, column=6, sticky=W)

#detect center of mass countor
def centroid(max_contour):
    if max_contour is not None:
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return 0, 0

#draw cirle in image
def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)

# configurete camera
def cameraConfig(threshold_distance):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    colorizer = rs.colorizer()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    clipping_distance_in_meters = threshold_distance
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color

    return pipeline, align_to, config, clipping_distance

#main void
def print_hi(name):
    pipeline, align_to, config, clipping_distance = cameraConfig(4.0)#configurate camera with threhowl 4 meters
    align = rs.align(align_to)
    pipe_profile = pipeline.start(config)
    calibrateY1 = calibrateY2 = calibrateY3 = calibrateY4 = calculate_screen = False
    #get intristic parameters
    intr = pipe_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = intr.fx
    fy = intr.fy
    px = intr.ppx
    py = intr.ppy
    intristic = np.zeros((3,3))
    print(intr)
    intristic[0][0] = fx
    intristic[1][1] = fy
    intristic[0][2] = px
    intristic[1][2] = py
    intristic[2][2] = 1
    print(intristic)
    X1_ = []
    Y1_ = []
    X2_ = []
    Y2_ = []
    X3_ = []
    Y3_ = []
    X4_ = []
    Y4_ = []
    y1_calibration = y2_calibration = y3_calibration = 0
    d1_calibration = d2_calibration = d3_calibration = 0
    scale = 0
    paint_go = False

    whiteScreenWeight = 850
    whiteScreenHeight = 500
    #drawing start screen with calibration point and canvas wint touch
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(str(whiteScreenWeight) + "x" + str(whiteScreenHeight) + "+" + str(screen_width - 900) + "+" + str(
        screen_height - 600))
    root.overrideredirect(-1)
    root.resizable(width=False, height=False)
    myCanvas = Canvas(root, width=whiteScreenWeight, height=whiteScreenHeight, bg='white')

    myCanvas.pack()
    # myCanvas.create_polygon()
    oval = myCanvas.create_oval(0, 0, 15, 15, outline="#f11", fill="#1f1", width=2)
    widget = Label(myCanvas, text='Touch', fg='white', bg='black')
    widget.pack()
    myCanvas.create_window(40, 10, window=widget)
    # myCanvas.create_oval(0,0, 80, 80)
    smoothing_filter = np.zeros(16)
    X_Prime = []
    Y_Prime = []
    counter_calib = 0
    calibration_points = []

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while 1:
            #get frame from camera
            frames = pipeline.wait_for_frames()
            #combine depth and image
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('1 - Color image',color_image)
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            #remove background
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('1 - DepthImage',depth_image)
            cv2.imshow('1 - Depth image',depth_colormap)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.imshow('2 - 3D - Remove background', bg_removed)  # limit distance
            image_hight, image_width, _ = bg_removed.shape

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('2 - Convert image from BGR to RGB',color_image)
            results = hands.process(color_image)
            distance = 0

            root.update_idletasks()
            root.update()

            #drawing points and click detection
            if results.multi_hand_landmarks and not calculate_screen:
                mp_drawing.draw_landmarks(color_image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight  # Указательный

                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))

                d = aligned_depth_frame.get_distance(int(x), int(y))
                draw_circles(color_image, [(int(x), int(y))])
                # print(x, y, d, distance)
            # cv2.imshow('colorimage', color_image)
            # print(distance)

            if results.multi_hand_landmarks and not calibrateY1:
                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight  # Указательный
                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))
                dis = aligned_depth_frame.get_distance(int(x), int(y))
                print(distance)
                if 4 < distance < 16:
                    # y1_calibration = aligned_depth_frame.get_distance(int(x), int(y))
                    d1_calibration = x
                    calibrateY1 = True
                    calibration_points.append([int(x), int(y), aligned_depth_frame.get_distance(int(x), int(y))])
                    time.sleep(1)
                    continue

            if results.multi_hand_landmarks and calibrateY1 and not calibrateY2:
                myCanvas.delete("all")
                myCanvas.create_oval(whiteScreenWeight - 15, 0, whiteScreenWeight, 15, outline="#f11", fill="#1f1",
                                     width=2)
                widget = Label(myCanvas, text='Touch', fg='white', bg='black')
                widget.pack()
                myCanvas.create_window(whiteScreenWeight - 60, 10, window=widget)
                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight  # Указательный
                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))
                dis = aligned_depth_frame.get_distance(int(x), int(y))
                distance = distance * dis
                print(distance)


                if 4 < distance < 20:
                    # y2_calibration = aligned_depth_frame.get_distance(int(x), int(y))
                    d2_calibration = x
                    calibrateY2 = True
                    calibration_points.append([int(x), int(y), aligned_depth_frame.get_distance(int(x), int(y))])
                    time.sleep(1)
                    continue

            if results.multi_hand_landmarks and calibrateY1 and calibrateY2 and not calibrateY3:
                myCanvas.delete("all")
                myCanvas.create_oval(whiteScreenWeight - 15, whiteScreenHeight - 15, whiteScreenWeight,
                                     whiteScreenHeight, outline="#f11", fill="#1f1", width=2)
                widget = Label(myCanvas, text='Touch', fg='white', bg='black')
                widget.pack()
                myCanvas.create_window(whiteScreenWeight - 60, whiteScreenHeight - 10, window=widget)

                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight  # Указательный
                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))
                dis = aligned_depth_frame.get_distance(int(x), int(y))
                distance = distance * dis
                print(distance)

                if 4< distance < 16:
                    # y3_calibration = aligned_depth_frame.get_distance(int(x), int(y))
                    calibration_points.append([int(x), int(y), aligned_depth_frame.get_distance(int(x), int(y))])
                    d3_calibration = x
                    calibrateY3 = True
                    time.sleep(1)

                    continue

            if results.multi_hand_landmarks and calibrateY1 and calibrateY2 and calibrateY3 and not calibrateY4:
                myCanvas.delete("all")
                myCanvas.create_oval(15, whiteScreenHeight - 15, 0, whiteScreenHeight,
                                     outline="#f11", fill="#1f1", width=2)
                widget = Label(myCanvas, text='Touch', fg='white', bg='black')
                widget.pack()
                myCanvas.create_window(60, whiteScreenHeight - 10, window=widget)
                myCanvas.create_window(whiteScreenWeight - 60, whiteScreenHeight - 10, window=widget)
                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight  # Указательный
                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))
                dis = aligned_depth_frame.get_distance(int(x), int(y))
                distance = distance * dis
                print(distance)

                if 4< distance < 12:
                    # y3_calibration = aligned_depth_frame.get_distance(int(x), int(y))
                    calibration_points.append([int(x), int(y), aligned_depth_frame.get_distance(int(x), int(y))])
                    d3_calibration = x
                    calibrateY4 = True
                    calculate_screen = True
                    time.sleep(1)
                    paint_go = True
                    print(calibration_points)
                    continue

            if paint_go:
                myCanvas.delete("all")
                root = Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.geometry(
                    str(whiteScreenWeight) + "x" + str(whiteScreenHeight) + "+" + str(screen_width - 900) + "+" + str(
                        screen_height - 600))
                root.overrideredirect(1)
                root.resizable(width=False, height=False)
                app = Paint(root)
                paint_go = False
            if results.multi_hand_landmarks and calculate_screen:
                root.update_idletasks()
                root.update()
                # color_image = color_image[0:330,:,:]

                mp_drawing.draw_landmarks(color_image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                x = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.WRIST].x * image_width  # Указательный
                y = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.WRIST].y * image_hight  # Указательный
                x_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width  # Указательный 1-я фаланга
                y_mcp = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_hight  # Указательный 1-я фаланга
                x_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].x * image_width  # Большой
                y_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP].y * image_hight  # Большой
                distance = abs(math.hypot(x_mcp - x_tip, y_mcp - y_tip))

                dis = aligned_depth_frame.get_distance(int(np.math.floor(x)), int(np.math.floor(y)))
                x1 = (calibration_points[0][0] * calibration_points[0][2] - px * calibration_points[0][2]) / fx
                y1 = (calibration_points[0][1] * calibration_points[0][2] - py * calibration_points[0][2]) / fy
                z1 = calibration_points[0][2]

                x2 = (calibration_points[1][0] * calibration_points[1][2] - px * calibration_points[1][2]) / fx
                y2 = (calibration_points[1][1] * calibration_points[1][2] - py * calibration_points[1][2]) / fy
                z2 = calibration_points[1][2]
                x3 = (calibration_points[2][0] * calibration_points[2][2] - px * calibration_points[2][2]) / fx
                y3 = (calibration_points[2][1] * calibration_points[2][2] - py * calibration_points[2][2]) / fy
                z3 = calibration_points[2][2]
                x4 = (calibration_points[3][0] * calibration_points[3][2] - px * calibration_points[3][2]) / fx
                y4 = (calibration_points[3][1] * calibration_points[3][2] - py * calibration_points[3][2]) / fy

                z4 = calibration_points[3][2]

                xnow = (x * dis - px * dis) / fx
                ynow = (y * dis - py * dis) / fy

                deltax = xnow - x1
                deltay = dis - z1

                scaleY = 1275 / abs(z1 - z3)
                scaleX = 750 / abs(x1 - x4)
                d = round(aligned_depth_frame.get_distance(int(np.floor(x)), int(np.floor(y))), 2)
                smoothing_filter = np.append(smoothing_filter,d)
                smoothing_filter = smoothing_filter[1:15]

                if min(smoothing_filter) > 0:
                    d = np.mean(smoothing_filter)

                if d > 0.15:
                    mouse.move(560 + deltay*scaleX,-deltax*scaleY*0.5 + 240)
                    print(560 + deltay*scaleX,-deltax*scaleY*0.5 + 240)
                distance = distance * dis
                print(distance)
                if 3 < distance < 15:
                    mouse.press(button='left')
                    # print('click')
                else:
                    mouse.release(button='left')

                cv2.imshow('colorimage', color_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
