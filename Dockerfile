FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install tensorflow
RUN pip3 install keras

COPY . .

CMD ["python3", "main.py"]