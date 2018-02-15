#-*- conding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, json, cv2, time
import numpy as np  
from fileinput import filename


subscription_key = '17c4484149cb45169ec7dc7a3d42ce0e'


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
capture = cv2.VideoCapture(0) 

if capture.isOpened() == False :
    print ('can not open CAM', CAM_ID)
    exit()

prevTime = 0

while(1):
    ret,frame = capture.read()  
    
    curTime = time.time()
    start_curtTime = time.strftime("[%y%m%d] %X",time.localtime())
    
    sec = curTime - prevTime
    prevTime = curTime
    
    fps  = 1/(sec)
    
    c = cv2.waitKey(25)& 0xFF
    count +=1 
    
    ##print ("Time %d" , format(sec))
    ##print ("Extimated fps %d" , format(fps))
    
    str = "FPS : %0.1f" % fps
    
   ## cv2.putText(frame, str, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    
    cv2.imshow( 'webcam', frame )
   ## fps = capture.get(cv2.CAP_PROP_FPS)
    if(count%10==0):
    ##print('save frame number : ' + str(int(capture.get(1))))
        cv2.imwrite("image%d.png" % 0, frame , params=[cv2.IMWRITE_PNG_COMPRESSION,0])

        
   ## print('fps', fps)    
       
    ##time.sleep(0.25)
    
    if(count%20 ==0):
        body = ""
        filename = ('D:/test/test/image%d.PNG' %0)

        f = open(filename,"rb")
        body = f.read()
        f.close

        try:
   
            response = requests.request('POST', uri_base + '/face/v1.0/detect', json=None, data=body, headers=headers, params=params)
            print ('Response:')
            parsed = json.loads(response.text)
            print (json.dumps(parsed, sort_keys=True, indent=2))
            
            
        except Exception as e:
            print('Error:')
            print(e)
            
    
    if c == 27:  
        break;  
    
    
capture.release()  
cv2.destroyAllWindows()  
