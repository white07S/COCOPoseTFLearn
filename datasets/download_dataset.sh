#!/bin/bash

download_and_extract() {
    local dataset_url=$1
    local zip_file=$(basename "$dataset_url")
    local folder_name="${zip_file%.zip}"

    # Check if the dataset folder already exists
    if [ -d "$folder_name" ]; then
        echo "The folder $folder_name already exists. Skipping download and extraction."
        return
    fi

    echo "Downloading ${zip_file}..."
    # Download the dataset
    curl -O "$dataset_url" || wget "$dataset_url"

    echo "Extracting ${zip_file}..."
    # Extract the dataset
    unzip -q "$zip_file"

    # Remove the zip file after extraction
    rm "$zip_file"

    echo "${zip_file} has been downloaded and extracted."
}

# URLs of the datasets
declare -a datasets=(
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
    "http://images.cocodataset.org/zips/test2017.zip"
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# Download and extract each dataset
for url in "${datasets[@]}"; do
    download_and_extract "$url"
done

echo "All datasets have been processed."
