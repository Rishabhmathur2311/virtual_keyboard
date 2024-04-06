from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
import math
from time import sleep
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)

cap.set(3, 1000)
cap.set(4, 720)

keys=[["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
      ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
      ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
      ["Z", "X", "C", "V", "B", "N", "M", ".", ",", "/"],
      [" "]
      ]



class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                '''

                    '''
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmlist

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length


detector = handDetector()

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 6, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    return img


class Button():
    def __init__(self, pos, text, size):
        self.pos=pos
        self.size=size
        self.text=text

    # def draw(self, img):


buttonList = []
def main():
    pTime = 0
    cTime = 0
    # cap = cv2.VideoCapture(0)
    detector = handDetector()

    length = len(keys) - 1
    finalText = ""
    keyboard = Controller()

    for i in range(len(keys) - 1):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([60 * j + 20, 60 * i + 20], key, [40, 50]))

    for j, key in enumerate(keys[length]):
        buttonList.append(Button([60 * j + 60, 270], key, [500, 50]))


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        img=drawAll(img, buttonList)

        # myButton.draw(img)

        if lmList:
            print(lmList[8][1])

            
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x<lmList[8][1]<x+w and y<lmList[8][2]<y+h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 6, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                    l=detector.findDistance([lmList[8][0], lmList[8][1]], [lmList[12][0], lmList[12][1]], img)

                    print(l)

                    if l<30:
                        keyboard.press(button.text)
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 175, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 6, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        finalText+=button.text
                        sleep(.5)

        cv2.rectangle(img, (50, 350), (600, 450), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, finalText, (60, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()