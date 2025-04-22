# use a python 3.11-slim image
FROM python:3.11-slim

# set the working directory inside the container
WORKDIR /app

# copy the requirements.txt file to the container
COPY requirements.txt .

# install dependencies from requirements.txt
RUN pip install -r requirements.txt

# copy the rest of the application
COPY . .

# expose port 5000 for Flask
EXPOSE 5000

# run the app
CMD ["python", "app.py"]
