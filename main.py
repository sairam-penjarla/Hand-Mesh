import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
hand_mpHands = mp.solutions.hands
hand_hands = hand_mpHands.Hands()
hand_mpDraw = mp.solutions.drawing_utils
p1,p2=((0,0),(0,0))
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hand_results = hand_hands.process(imgRGB)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                tips = [0,4,8,12,16,20]
                if id in tips:
                    cv2.circle(img,(cx,cy),15,(255,255,255),cv2.FILLED)
            hand_mpDraw.draw_landmarks(
                img,
                handLms,
                hand_mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec =  hand_mpDraw.DrawingSpec(color=(0,0,0)),
                connection_drawing_spec =  hand_mpDraw.DrawingSpec(color=(201, 194, 2))
            )
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
