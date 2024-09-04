# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /var/flasksite

# Copy only the requirements file first
COPY requirements.txt /var/flasksite/

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . /var/flasksite/

# Expose the port Flask will run on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the application
CMD ["flask", "run"]
