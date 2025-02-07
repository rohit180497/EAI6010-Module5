# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application source code
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Run the application using Gunicorn (Point to `app.py`)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
