#-*- conding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, json, cv2, time
import numpy as np  
from fileinput import filename
import dis
from socket import *



subscription_key = '552230d5ad9e4f9a8b31f1d1ea44ab42'


uri_base = 'https://westcentralus.api.cognitive.microsoft.com'


headers = {
    'Content-Type': 'application/ octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}


params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

count = 0
CAM_ID = 0

anger_a = 0
contempt_a = 0
disgust_a  = 0
fear_a = 0
happiness_a = 0
neutral_a = 0
sadness_a = 0
surprise_a = 0

capture = cv2.VideoCapture(CAM_ID) 

if capture.isOpened() == False :
    print ('can not open CAM', CAM_ID)
    exit()

prevTime = 0



TRACKING_STATE_CHECK=0

TRACKING_STATE_INIT =1

TRACKING_STATE_ON = 2


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
   
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
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

    
    TrackingState = 0
    
    TrackingROI = (0,0,0,0)


while(1):
    
    ref,frame = capture.read()  
    
    curTime = time.time()
    
    
    sec = curTime - prevTime
    prevTime = curTime
    
    
    fps  = 1/(sec)
    
    c = cv2.waitKey(25)& 0xFF
    count +=1 
    
    ##print ("Time %d" , format(sec))
    ##print ("Extimated fps %d" , format(fps))
    
    if TrackingState == TRACKING_STATE_CHECK:
            
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            grayframe = cv2.equalizeHist(grayframe)
           
            faces  = face_cascade.detectMultiScale(grayframe, 1.1, 5, 0, (30, 30))

            if len(faces) > 0:
                
                x,y,w,h = faces[0]
                
                TrackingROI = (x,y,w,h)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3, 4, 0)
                
                TrackingState = TRACKING_STATE_INIT
                print('det w : %d ' % w + 'h : %d ' % h)


      
    elif TrackingState == TRACKING_STATE_INIT:
            
            ref = tracker.init(frame, TrackingROI)
            if ref:
                
                TrackingState = TRACKING_STATE_ON
                print('tracking init succeeded')
            else:
                
                TrackingState = TRACKING_STATE_CHECK
                print('tracking init failed')

        
    elif TrackingState == TRACKING_STATE_ON:
           
            ref, TrackingROI = tracker.update(frame)
            if ref:
                
                p1 = (int(TrackingROI[0]), int(TrackingROI[1]))
                p2 = (int(TrackingROI[0] + TrackingROI[2]), int(TrackingROI[1] + TrackingROI[3]))
               
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                print('success x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) +
                        'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))
            else:
                print('Tracking failed')

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
    
    cv2.putText(frame, str, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_n, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_a, (0,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_f, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_c, (0,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_d, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_h, (0,175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_s, (0,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(frame, str_ss, (0,225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
   ## gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow( 'webcam', frame )
   ## fps = capture.get(cv2.CAP_PROP_FPS)
    ##if(count%20==0):
    ##print('save frame number : ' + str(int(capture.get(1))))
    cv2.imwrite("image%d.png" % 0, frame , params=[cv2.IMWRITE_PNG_COMPRESSION,0])

        
   ## print('fps', fps)    
       
    ##time.sleep(0.25)
    
    if(count%50 ==0):
        body = ""
        filename = ('C:/image/test2/test/image%d.PNG' %0)

        f = open(filename,"rb")
        body = f.read()
        f.close

        try:
   
            response = requests.request('POST', uri_base + '/face/v1.0/detect', json=None, data=body, headers=headers, params=params)
            print ('Response:')
            parsed = json.loads(response.text)
            print (json.dumps(parsed, sort_keys=True, indent=2))
           ## print(type(parsed))
            aa = parsed.pop(0)
            
            ##print (type(aa))
            emo = aa.pop('faceAttributes')
            emotion_a = emo.pop('emotion')
            
            anger_a = emotion_a['anger']
            contempt_a = emotion_a['contempt']
            disgust_a  = emotion_a['disgust']
            fear_a = emotion_a['fear']
            happiness_a = emotion_a['happiness']
            neutral_a = emotion_a['neutral']
            sadness_a = emotion_a['sadness']
            surprise_a = emotion_a['surprise']
            
            
           ## print (anger_a)
            ##print (neutral_a)
            ##print (fear_a)
           
            ##print (type(neutral_a))
            ##print (type(emotion_a['neutral']))
            
            
            
            
        except Exception as e:
            print('Error:')
            anger_a = 0.000
            contempt_a = 0.000
            disgust_a  = 0.000
            fear_a = 0.000
            happiness_a = 0.000
            neutral_a = 0.000
            sadness_a = 0.000
            surprise_a = 0.000
            
    
    
    if c == 27:  
        break;  

    
capture.release()  
cv2.destroyAllWindows()  