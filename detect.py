import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame()
letters = list()
accuracy = list()
frames = list()
mesaj = list()
def run():
    model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5\best.pt', source='local')
    sentence2 = ''
    cap = cv2.VideoCapture(0)
    count = 1
    while cap.isOpened():
        success, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF
        results = model(frame)
        font = cv2.FONT_HERSHEY_PLAIN
        # cv2.imshow('YOLO', np.squeeze(results.render()))
        res = results.pandas().xyxy[0]['name']
        acc = results.pandas().xyxy[0]['confidence']
        if len(res) and acc[0]:
            res = res[0]
            acc = round(acc[0],2)
            print(res, acc)
            if key == 13:
                sentence2 += res
                print(sentence2)
            text = "now : "+res+" "+str(acc)
            accuracy.append(acc)
            letters.append(res)
            frames.append(frame)
            mesaj.append(sentence2)
            cv2.putText(frame, text, (500, 20), font, 1, (222, 222, 222), 1)
        cv2.putText(frame, sentence2, (80, 420), font, 2, (222, 222, 222), 2)
        cv2.imshow('Detectarea literelor alfabetului American.', frame)
        cv2.imwrite("images\\frame%d.jpg" % count, frame)
        count += 1

        if key == ord('q'):
            break;
    df["frames"], df["class"], df["confidence"], df["mesaj"] = frames, letters, accuracy, mesaj
    df.to_csv('results.csv')
    cap.release()
    cv2.destroyAllWindows()
run()
