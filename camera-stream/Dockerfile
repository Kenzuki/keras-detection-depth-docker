FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get -y install libusb-1.0-0
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install pyrealsense2
RUN pip3 install websockets
RUN pip3 install Pillow

WORKDIR /app/

COPY . /app/

EXPOSE 5678

CMD ["python3", "-u", "/app/main.py"]