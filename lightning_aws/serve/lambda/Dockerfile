# Pull the base image with python 3.9 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.9

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.9 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

COPY model.pt ./

CMD ["app.handler"]