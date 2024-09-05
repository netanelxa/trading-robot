FROM python:3.9-slim

RUN mkdir /var/flasksite

COPY . /var/flasksite/

WORKDIR /var/flasksite/

RUN pip3 install -r requirements.txt
RUN opentelemetry-bootstrap --action=install


EXPOSE 5000

# TODO: move to k8s config-map
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

#ENV OTEL_RESOURCE_ATTRIBUTES=service.name=flaskApp 
#ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
#ENV OTEL_EXPORTER_OTLP_PROTOCOL=grpc
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true

# CMD ["flask", "run"]
CMD [ "opentelemetry-instrument", "--traces_exporter",  "console", "--service_name", "web-ui", "flask", "run" ]
