# Use an official Python runtime as a parent image
FROM python:3.10.14-slim-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY ./src /usr/src/app/src
COPY ./data /usr/src/app/data
COPY ./requirements.txt /usr/src/app

RUN apt-get update && apt-get -y install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]
