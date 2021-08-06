#FROM python:3.8-slim
FROM centos/python-36-centos7

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY /codigo/requirements .

# install dependencies
RUN pip install -r requirements

# copy the content of the local src directory to the working directory
COPY codigo/ .

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "80"]
