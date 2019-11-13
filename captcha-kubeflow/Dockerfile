# tensorflow images
FROM tensorflow/tensorflow:1.15.0

# Install python3
RUN  apt-get install -y python3

# Install pip
RUN apt-get install -y wget vim

# Install cmake
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y cmake

# Install OpenCV
RUN apt-get install -y unzip
RUN wget -P /usr/local/src/ https://github.com/opencv/opencv/archive/4.1.1.zip
RUN cd /usr/local/src/ && unzip 4.1.1.zip && rm 4.1.1.zip
RUN cd /usr/local/src/opencv-4.1.1/ && mkdir build
RUN cd /usr/local/src/opencv-4.1.1/build && cmake -D CMAKE_INSTALL_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ .. && make -j4 && make install

# Install opencv-python
RUN pip install opencv-python
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev

ADD . /var/tf_captcha
ENTRYPOINT ["python", "/var/tf_captcha/dist-captcha.py"]

