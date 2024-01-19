# Use an official base image with Miniconda installed
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the conda_requirements.txt and requirements.txt files into the container
COPY conda_requirements.txt .
COPY requirements.txt .

# Create the Conda environment using your conda_requirements.txt
RUN conda create --name cv --file conda_requirements.txt

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cv", "/bin/bash", "-c"]

# Install additional requirements using pip from your requirements.txt
RUN pip install -r requirements.txt

# Copy the training directory into the container
COPY training/ ./training

# Set the default command to execute
# when creating a new container with this image
CMD ["conda", "run", "-n", "cv", "python", "training/train_pose.py"]
