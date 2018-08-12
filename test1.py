# -*- conding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, json, cv2, time
import numpy as np
from fileinput import filename
import dis
from socket import *
import simplejson

subscription_key = '79bdaf61a968450caaac4cd3005bb16a'

uri_base = 'https://eastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = {
    'Content-Type': 'application/ octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

count = 0
CAM_ID = 0

anger_a = 0
contempt_a = 0
disgust_a = 0
fear_a = 0
happiness_a = 0
neutral_a = 0
sadness_a = 0
surprise_a = 0

trackWindow = None
roi_hist = None



capture = cv2.VideoCapture(CAM_ID)

if capture.isOpened() == False:
    print('can not open CAM', CAM_ID)
    exit()

prevTime = 0
ccount = 0

TRACKING_STATE_CHECK = 0

TRACKING_STATE_INIT = 1

TRACKING_STATE_ON = 2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    global term_crit

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.MultiTracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    face_cascade = cv2.CascadeClassifier()

    face_cascade.load('C:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    ##face_cascade.load('C:/image/test2/cascade2.xml')

    TrackingState = 0

    TrackingROI = (0, 0, 0, 0)


while (1):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    ref, frame = capture.read()

    curTime = time.time()

    sec = curTime - prevTime
    prevTime = curTime

    fps = 1 / (sec)

    c = cv2.waitKey(25) & 0xFF
    count += 1




    if TrackingState == TRACKING_STATE_CHECK:
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grayframe = cv2.equalizeHist(grayframe)

        faces = face_cascade.detectMultiScale(grayframe, 1.8, 2, 0, (30, 30))


        if len(faces) > 0:
            x, y, w, h = faces[0]

            TrackingROI = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3, 4, 0)

            TrackingState = TRACKING_STATE_INIT


    elif TrackingState == TRACKING_STATE_INIT:

        ref = tracker.init(frame, TrackingROI)
        if ref:

            TrackingState = TRACKING_STATE_ON
            print('tracking succeeded')
        else:

            TrackingState = TRACKING_STATE_ON


    elif TrackingState == TRACKING_STATE_ON:

        ref, TrackingROI = tracker.update(frame)
        if ref:

            p1 = (int(TrackingROI[0]), int(TrackingROI[1]))
            p2 = (int(TrackingROI[0] + TrackingROI[2]), int(TrackingROI[1] + TrackingROI[3]))

            cv2.rectangle(frame, p1, p2, (0, 0, 255), 1, 1)
            print('face x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) +
                  'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))


            trackWindow = (x, y, w, h)

            roi = frame[x:x + h, y:y + w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, trackWindow = cv2.CamShift(dst, trackWindow, term_crit)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame, [pts], True, 255, 2)

        else:
            print('Tracking failed')
            ccount = ccount + 1

            TrackingState = TRACKING_STATE_CHECK

    str = " FPS : %0.3f" % fps
    str_n = " neutral : %0.3f" % neutral_a
    str_a = " anger : %0.3f" % anger_a
    str_f = " fear : %0.3f" % fear_a
    str_c = " contempt : %0.3f" % contempt_a
    str_d = " disgust : %0.3f" % disgust_a
    str_h = " happy : %0.3f" % happiness_a
    str_s = " sad : %0.3f" % sadness_a
    str_ss = " surprise : %0.3f" % surprise_a

    cv2.putText(frame, str, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_n, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_a, (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_f, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_c, (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_d, (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_h, (0, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_s, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(frame, str_ss, (0, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow('webcam', frame)

    cv2.imwrite("image%d.png" % 0, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])


    if (count % 50 == 0):
        body = ""
        filename = ('C:/image/test2/image%d.PNG' % 0)

        f = open(filename, "rb")
        body = f.read()
        f.close

        try:


            response= requests.post(uri_base, data=body, headers=headers, params=params)


            print('Response:')
            parsed = json.loads(response.text)
            print(json.dumps(parsed, sort_keys=True, indent=2))
            aa = parsed.pop(0)

            emo = aa.pop('faceAttributes')
            fac = aa.pop('faceRectangle')

            emotion_a = emo.pop('emotion')

            anger_a = emotion_a['anger']
            contempt_a = emotion_a['contempt']
            disgust_a = emotion_a['disgust']
            fear_a = emotion_a['fear']
            happiness_a = emotion_a['happiness']
            neutral_a = emotion_a['neutral']
            sadness_a = emotion_a['sadness']
            surprise_a = emotion_a['surprise']


            height = fac.pop['height']
            width = fac.pop['width']
            col = fac.pop['left']
            row = fac.pop['top']


        except Exception as e:
            print('Error:')
            anger_a = 0.000
            contempt_a = 0.000
            disgust_a = 0.000
            fear_a = 0.000
            happiness_a = 0.000
            neutral_a = 0.000
            sadness_a = 0.000
            surprise_a = 0.000

    if c == 27:
        break;

capture.release()
cv2.destroyAllWindows()  