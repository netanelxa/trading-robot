FROM python:3.9-slim

RUN mkdir /var/flasksite

COPY . /var/flasksite/

WORKDIR /var/flasksite/

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]
