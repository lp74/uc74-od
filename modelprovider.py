import os
import six.moves.urllib as urllib
import tarfile

class ModelProvider:

  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

  modelNames = {
    'ssd_mobilenet_v1_coco': 'ssd_mobilenet_v1_coco_2018_01_28',                        #
    'ssd-mobilenet' : 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',     #
    'ssd-mobilnet-v2-coco': 'ssd_mobilenet_v2_coco_2018_03_29',                         # rate: * * * * o
    'faster_rcnn_inception_v2_coco': 'faster_rcnn_inception_v2_coco_2018_01_28',        # 
    'faster_rcnn_resnet50_coco': 'faster_rcnn_resnet50_coco_2018_01_28'                 #
    }

  def __init__(self, key):
    self.key = key
  
  def getKey(self):
    return self.key

  def get(self):
    if not(os.path.isfile(self.modelFile(self.key))):
      self.download(self.key)
    return self.modelFile(self.key)
  
  def download(self, key):
    print('Downloading the model')
    
    opener = urllib.request.URLopener()
    opener.retrieve(self.DOWNLOAD_BASE + self.tarfile(key), self.tarfile(key), reporthook=self.showProgress)
    
    tar_file = tarfile.open(self.tarfile(key))
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    
    print('Model downloaded')
  
  def tarfile(self, key):
    return self.modelName(key) + '.tar.gz'

  def showProgress(self, a, b, c):
    print(a, b, round(a/float(b)*100,1), "% \r"),

  def modelName(self, key):
    return self.modelNames[key]
  
  def modelFile(self, key):
    return self.modelName(key) + '/frozen_inference_graph.pb'