# Use an official base image with Miniconda installed
FROM continuumio/miniconda3

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the conda_requirements.txt and requirements.txt files into the container
COPY conda_requirements.txt .
COPY requirements.txt .

RUN conda config --set ssl_verify false

# Create the Conda environment using your conda_requirements.txt
RUN conda create --name cv --file conda_requirements.txt

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cv", "/bin/bash", "-c"]

# Install additional requirements using pip from your requirements.txt
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the scripts directory into the container and run the download_dataset.sh script
COPY scripts/ ./scripts
RUN chmod +x ./scripts/download_dataset.sh && ./scripts/download_dataset.sh

# Copy the training directory into the container
COPY training/ ./training