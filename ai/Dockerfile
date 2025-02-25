FROM python:3.13

WORKDIR /app
COPY requirements.txt .

RUN apt update
RUN apt install libopenmpi-dev openmpi-bin libopenmpi3 -y

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]
