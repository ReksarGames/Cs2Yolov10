import argparse
import time
from threading import Thread

import numpy as np
from pynput import keyboard, mouse

from args_ import *
from utils.controls import listen
from utils.controls.listen import listen_k_press, listen_k_release, listen_m_click, listen_init, get_D_L, \
    mouse_redirection, move_mouse
from utils.grabber import capture
from utils.grabber.capture import *
from utils.grabber.draw import show_target

global detecting, listening

target_classes = [1, 2]

def listen_t_press(key):
    global target_classes
    try:
        if key.char == 'k':
            # Переключаем классы целей при нажатии 'k'
            if target_classes == [2]:
                target_classes = [1]  # Меняем на [1]
                print("Target classes switched to [1]")
            else:
                target_classes = [2]  # Возвращаем обратно [2]
                print("Target classes switched to [2]")
        if key.char == 'p':
            target_classes = [1, 2]  # Присваиваем [1,2] при нажатии 'p'
            print("Target classes switched to [1, 2]")

    except AttributeError:
        pass


def listeners():
    keyboard_listener = keyboard.Listener(on_press=listen_k_press, on_release=listen_k_release)
    keyboard_listener.start()

    keyboard_listener = keyboard.Listener(on_press=listen_t_press)
    keyboard_listener.start()
    keyboard_listener.join()


    mouse_listener = mouse.Listener(on_click=listen_m_click)
    mouse_listener.start()

    keyboard_listener.join()

if __name__ == "__main__":
    os.system("")
    print("\033[01;04;31m" + "A" + "\033[32m" + "N" + "\033[33m" + "S" + "\033[34m" + "I" + "\033[00m" + " enabled")
    # create an arg set
    listening = True
    print("listeners start")

    args = argparse.ArgumentParser()
    args = arg_init(args)
    listen_init(args)

    thread_1 = Thread(target=listeners)
    thread_1.start()
    print(thread_1)

    capture_init(args)
    if args.model[-3:] == ".pt":
        from utils.grabber.predict import *

        predict_init(args)

    print("main start")
    time_start = time.time()
    count = 0
    time_capture_total = 0
    while listening:

        detecting, listening = get_D_L()
        # take a screenshot
        time_shot = time.time()
        img = take_shot(args)
        time_capture = time.time()
        time_capture_total += time_capture - time_shot
        # predict the image
        time.sleep(args.wait)
        if args.model[-3:] == ".pt":
            predict_output = predict(args, img)
            # print(predict_output.boxes.cls)

            # Update: filter boxes by class indices 1 and 2 head
            boxes = predict_output.boxes
            boxes = boxes[np.isin(boxes.cls.cpu().numpy(), target_classes)].cpu().xyxy.numpy()

        time_predict = time.time()

        if detecting:
            if boxes.size != 0:
                if args.draw_boxes:
                    for i in range(0, int(boxes.size/4)):
                        show_target([int(boxes[i,0]) + capture.x, int(boxes[i,1]) + capture.y, int(boxes[i,2]) + capture.x, int(boxes[i,3]) + capture.y])

            mouse_redirection(args, boxes)
            move_mouse(args)

        count += 1

        if (count % 100 == 0):
            time_per_100frame = time.time() - time_start
            time_start = time.time()
            print("Screenshot fps: ", count / time_capture_total)
            print("fps: ", count / time_per_100frame)
            interval = time_per_100frame / count
            print("interval: ", interval)
            print("[LEFT_LOCK]" if listen.left_lock else "[         ]", "[RIGHT_LOCK]" if listen.right_lock else "[          ]", "[\033[30;41mAUTO_FIRE\033[00m]" if listen.auto_fire else "[          ]")
            count = 0
            time_capture_total = 0

    print("main stop")
