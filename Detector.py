from collections import defaultdict
import numpy as np
import os
import sys
import tensorflow as tf
import cv2

from logger import logger
from HTTPStreamReader import HTTPStreamReader
from modelprovider import ModelProvider

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("TF_PATH"))
sys.path.append(os.getenv("OD_PATH"))

model = sys.argv[1] or 'ssd'


from utils import label_map_util
#from utils import visualization_utils as vis_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.getenv("OD_PATH"), 'data', 'mscoco_label_map.pbtxt')


def load_image_into_numpy_array(image):
  return np.array(image).astype(np.uint8)

NUM_CLASSES = 90

class Detector:
  def __init__(self, model='ssd-2018'):
    self._model = model
    logger.info('Instantiating the Detector', self._model)
    currentModel = ModelProvider(model)
    self._detection_graph = tf.Graph()
    with self._detection_graph.as_default():  
      od_graph_def = tf.GraphDef()

      with tf.gfile.GFile(currentModel.get(), 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    categories = label_map_util.convert_label_map_to_categories(
      label_map, 
      max_num_classes = NUM_CLASSES, 
      use_display_name = True
      )
    
    self._category_index = label_map_util.create_category_index(categories)
    
    with self._detection_graph.as_default():
      config = tf.ConfigProto(log_device_placement=False) # True to view tensorflow details (i.e. GPU)
      config.intra_op_parallelism_threads = 0
      config.inter_op_parallelism_threads = 0
      self._sess = tf.Session(graph=self._detection_graph, config=config)
      self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
      self._detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
      self._detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
      self._detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
      self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

  def detect(self, np_image):
        try:
          image_np_expanded = np.expand_dims(np_image, axis=0)
          (boxes, scores, classes, num) = self._sess.run(
            [
              self._detection_boxes, 
              self._detection_scores, 
              self._detection_classes, 
              self._num_detections
            ], 
            feed_dict = {
              self._image_tensor: image_np_expanded
              })

          logger.info('Detected %d objects', num)
          for i in range (0, num.astype(int)[0]):
            current_class = self._category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
            current_score = np.squeeze(scores)[i]
            
            logger.info('class %s - score %.2f', current_class, current_score)
            
            if(current_class == u'person' and current_score >= 0.05 ):
              logger.info('MAN probability: %.2f', current_score)
            
            if(current_class == u'person' and current_score >= 0.05 ):
              cv2.imwrite('/Users/lp74/Desktop/face-recognition/unknown_pictures/man.jpg', np_image)
              os.system('/miniconda3/bin/face_recognition ~/Desktop/face-recognition/pictures_of_people_i_know/ ~/Desktop/face-recognition/unknown_pictures/man.jpg')
            
            if(current_class == u'dog' and current_score >= 0.05):
              logger.info('DOG probability: %.2f', current_score)
          
          return (boxes, scores, classes, num)
          
        except Exception as err:
          print(err)
          pass


detector = Detector(sys.argv[1])
clear = lambda : os.system('clear')
clear()

def resizer(scale_percent):
  def resize(jpg):
    cv2_image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    width = int(cv2_image.shape[1] * scale_percent / 100)
    height = int(cv2_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    cv2_resized = cv2.resize(cv2_image, dim, interpolation = cv2.INTER_AREA)
    return cv2_resized
  return resize

def template(jpg): 
  cv2_resized = resizer(100)(jpg)
  (boxes, scores, classes, num) = detector.detect(cv2_resized)
  box = boxes[0][0]
  print(box)
  H = cv2_resized.shape[0]
  W = cv2_resized.shape[1]
  print(W, H)
  cv2.rectangle(cv2_resized, (int(box[1]*W), int(box[0]*H)), (int(box[3]*W), int(box[2]*H)), (0, 255, 0), 2)
  cv2.imshow('hoststr', cv2_resized) 
  if cv2.waitKey(1) == 27:
      exit(0) 

# Instantiate the stream reader class
http_stream = HTTPStreamReader('http://192.168.0.9:8888')
# Operations: 1 - resize, 
http_stream.subscribe(template)
http_stream.start()
