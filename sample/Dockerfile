# Use the netanelxa/talib image as the base image
FROM netanelxa/talib:latest

# Set the working directory
WORKDIR /var/flasksite

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev

# Copy only the requirements file first
COPY requirements.txt /var/flasksite/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN opentelemetry-bootstrap --action=install

# Copy the rest of the application code
COPY . /var/flasksite/

# Expose the port Flask will run on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true

# Command to run the application
CMD [ "opentelemetry-instrument", "--traces_exporter",  "console", "--service_name", "web-ui", "flask", "run" ]
