FROM python:3
COPY . /app
WORKDIR /app
RUN pip install cmake
RUN pip install dlib
RUN pip install matplotlib
RUN pip3 install face_recognition
RUN pip install opencv-python
RUN pip install numpy
RUN unset DISPLAY XAUTHORITY
RUN xvfb-run python face_rec_harsh.py
CMD python ./face_rec_harsh.py
