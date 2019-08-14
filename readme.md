# Readme

## Tensorflow models
Follow [this instructions](https://github.com/tiangolo/tensorflow-models/blob/master/research/object_detection/g3doc/installation.md).

## Environment vars
Add a <code>.env</code> file with the following lines:

```bash
TF_PATH=/Users/lucapolverini/Development/models/research # change with your path
OD_PATH=/Users/lucapolverini/Development/models/research/object_detection/ # change with your path
```

## Streaming the camera MJPG

```
[Unit]
Description=Stream Service
After=network.target

[Service]
WorkingDirectory=/home/lucapolverini/
ExecStart=cvlc v4l2:///dev/video0:chroma=MJPG:width=640:height=360 --sout '#transcode{vcodec=MJPG,fps=1,threads=2,venc=ffmpeg{strict=1}}:standard{access=http{mime=multipart/x-mixed-replace;boundary=--7b3cc56e5f51db803f790dad720ed50a},mux=mpjpeg,dst=:8888}' -vvv
Restart=on-failure
User=lucapolverini

[Install]
WantedBy=multi-user.target
```