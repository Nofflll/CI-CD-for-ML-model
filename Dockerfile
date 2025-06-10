# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the dependencies file to the working directory
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code to the working directory
COPY ./src /app/src
COPY ./models /app/models
COPY ./params.yaml /app/params.yaml

# 6. Make port 8000 available to the world outside this container
EXPOSE 8000

# 7. Define environment variable
ENV NAME World

# 8. Run app.py when the container launches
# Use gunicorn for production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "src.app:app"] 