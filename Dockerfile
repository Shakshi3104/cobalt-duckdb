FROM python:3.11

# copy requirements
COPY requirements.txt requirements.txt

# install
RUN apt-get update && apt-get clean;
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libssl-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY .env .env
COPY model/ model/
# COPY view/ view/
COPY app.py app.py

ENV PYTHONUNBUFFERED 1

CMD gradio app.py