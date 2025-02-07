# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY app/ /app/

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["flask", "--app", "main.py", "run", "--host=0.0.0.0"]
