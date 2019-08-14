import six.moves.urllib as urllib

class HTTPStreamReader:
  def __init__(self, url):
    self._url = url
    self._observers = [];
  def subscribe(self, fn):
    self._observers.append(fn)
  def start(self):
    self._stream = urllib.request.urlopen(self._url)
    bytes = b''
    while True:
      try:
        bytes += self._stream.read(16384)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a!=-1 and b!=-1:
          jpg = bytes[a:b+2]
          bytes = bytes[b+2:]
          for fn in self._observers: 
            fn(jpg)
      except Exception as error:
        print(error)
        self._stream.close()
        pass 