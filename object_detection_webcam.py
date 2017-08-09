
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[12]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image


# ## Env setup

# In[13]:


# This is needed to display the images.
#get_ipython().magic(u'matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[14]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[15]:


# What model to download.

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1


# ## Download Model

# In[16]:

if MODEL_NAME not in os.listdir("./"):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[17]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[18]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[19]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



def filterForClass(classNum, classes, scores, boxes):
    #print(scores)
    # only choose class 1 (persons)
    classes = [classy for classy in np.squeeze(classes).astype(np.int32) if classy==classNum]
    scores = np.squeeze(scores)[:len(classes)]
    #only show good quality matches
    boxes = np.squeeze(boxes)[:len(classes)]
    return (classes,scores,boxes)

def addMousePointToPoolVertex(event, x, y, flags, param):
    global touchPoints
    # if left mouse button clicked, add to pool vertex array
    if event == cv2.EVENT_LBUTTONUP:
        refPt = (x, y)
        print("Pool vertex point added!")
        addPoolVertexPoint(touchPoints, refPt)
        
    #right click to clear points
    elif event == cv2.EVENT_RBUTTONUP:
        touchPoints = []
        
        
def addPoolVertexPoint(points, point):
    points.append(point)
    
def drawPoolEdges(points, image):
    npPoints = np.array(points)

    cv2.polylines(image,[npPoints],True,(255,255,0), 2)

def triggerPersonInPoolAlarm(withFrame, sendFrame):
    global lastAlertTime
    if time.time()> lastAlertTime + ALERT_SPACING_SECONDS:
        lastAlertTime = time.time()
        text = ("There's a person in your pool!")
        print(text)
#        
#        # Find these values at https://twilio.com/user/account
         #uncomment following lines to send texts: ANDREW
#        account_sid = "ACbbdd3feb51afefc8028653d491dbafe7"
#        auth_token = "35c127b78150a9a1792a3b5c2ffef972"
#        client = Client(account_sid, auth_token)
#        print("text is: "+ text)
#        message = client.api.account.messages.create (to="+17038511458",
#                                                     from_="+12403486466",
#                                                     body=text)
#        
#        #if you want to send a picture of the pool as well:
#        if sendFrame:
#            #imageCaption = "This is a picture of the person in your pool"
#            #record time
#            filename = "PoolAtTime" + str(int(time.time())) + ".jpg"
#            cv2.imwrite(filename, withFrame)    
#            messageWithFrame = client.api.account.messages.create (to="+17038511458",
#                                                     from_="+12403486466",
#                                                     body=text,
#                                                     media_url=NGROK_URL+filename
#                                                     )
        #GIVES ERROR: TypeError: the JSON object must be str, not 'bytes'
       

def personInPoolTest(poolPoints, personPoints):
    if len(poolPoints)>2:
        npTouchArray = np.array(poolPoints)
        if len(personPoints)==2: #(only has the top left and bottom right corners)
            personPoints=list(personPoints)
            personPoints.append((personPoints[0][0],personPoints[1][1])) #add top right
            personPoints.append((personPoints[1][0],personPoints[0][1])) # add bottom left
        for point in personPoints:
            if type(point)!=tuple:
                point=tuple(point)
            if cv2.pointPolygonTest(npTouchArray, point, False) != -1:
                return True
    return False    
  
def poolInPersonTest(poolPoints, personPoints):
    if len(poolPoints)>2:
        if len(personPoints)==2: #(only has the top left and bottom right corners)
            personPoints=list(personPoints)
            personPoints.append((personPoints[0][0],personPoints[1][1])) #add top right
            personPoints.append((personPoints[1][0],personPoints[0][1])) # add bottom left
        npTouchArray = np.array(personPoints).astype(np.int32)
#        print(npTouchArray)
        for point in poolPoints:
            if type(point)!=tuple:
                point=tuple(point)
            if cv2.pointPolygonTest(npTouchArray, point, False) != -1:
                return True
    return False    
  


# # Detection

# In[10]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)


# In[11]:
cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
FPS = 5
width,height = 1080, 720
#width, height = cap.get(3), cap.get(4)
#print(width,height, "are the w and h of the capture")
#print("Now setting it to {} fps: ".format(FPS) + str(cap.set(5, FPS)) )
#print('frame rate is currently: ', cap.get(5))

#Andrew: change this variable if it's identifying too many humans
MINIMUM_SCORE_THRESH=0.6
#ALERTS AND POOL VERTICES HERE 
touchPoints = [] 
ALERT_SPACING_SECONDS = 4
lastAlertTime = 0

cv2.namedWindow("webcam")
cv2.setMouseCallback("webcam", addMousePointToPoolVertex)

if cap.isOpened():
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
          while True:
                read, frame = cap.read()
                if read:
                    #for image_path in TEST_IMAGE_PATHS:
                      image_np = frame #Image.open(image_path)
                      # the array based representation of the image will be used later in order to prepare the
                      # result image with boxes and labels on it.
                      #image_np = image #load_image_into_numpy_array(image)
                      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                      image_np_expanded = np.expand_dims(image_np, axis=0)
                      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                      # Each box represents a part of the image where a particular object was detected.
                      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                      # Each score represent how level of confidence for each of the objects.
                      # Score is shown on the result image, together with the class label.
                      scores = detection_graph.get_tensor_by_name('detection_scores:0')
                      classes = detection_graph.get_tensor_by_name('detection_classes:0')
                      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                      # Actual detection.
                      
                      (boxes, scores, classes, num_detections) = sess.run(
                          [boxes, scores, classes, num_detections],
                          feed_dict={image_tensor: image_np_expanded})
           
                      #filter for humans 
                      classes, scores, boxes = filterForClass(1, classes, scores, boxes)
                      if len(boxes)<2:
                          widthCorrections = [point*width for point in box]     
                          heightCorrections = [point*height for point in box ]
                      else:
                      #multiply the points by the width and heights, tensorflow is location people based on a scale from 0 to 1
                          widthCorrections = [[point*width for point in box] for box in boxes]    
                          heightCorrections = [[point*height for point in box ] for box in boxes] 
                      #for some reason, the order of coordinates in the boxes is y,x,y,x
                      onlyYes = [y[::2] for y in heightCorrections]
                      onlyXes = [x[1::2] for x in widthCorrections]
                      locationBoxes=[]
                      for i in range(len(onlyXes)):
                          if scores[i]>MINIMUM_SCORE_THRESH:
                              locationBoxes.append(zip(onlyXes[i],onlyYes[i]))                      # Visualization of the results of a detection.
                      for box in locationBoxes:
                          
#                          cv2.rectangle(image_np, p1, p2, (255,255,255),3)
                          if personInPoolTest(touchPoints, box) or poolInPersonTest(touchPoints, box):
                          #print("there is a person in the pool!")
                          #trigger alarm: (comment to not receive texts), change bool to send pic in message
                              triggerPersonInPoolAlarm(image_np, False)
                    # ANDREW: may want to change minScore (default) variable
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          np.array(boxes),
                          np.array(classes).astype(np.int32),
                          (scores),
                          category_index,
                          use_normalized_coordinates=True,
                          line_thickness=8,
                          min_score_thresh=MINIMUM_SCORE_THRESH)
                      #plt.figure(figsize=IMAGE_SIZE)
                      #plt.imshow(image_np)
                      drawPoolEdges(touchPoints, image_np)

                      cv2.imshow("webcam",image_np) 
                      if cv2.waitKey(1) & 0xFF == ord('q'):
                           # release the capture
                           cap.release()
                           cv2.destroyAllWindows()
                           print("exiting operations complete")
                           break

# In[ ]:


