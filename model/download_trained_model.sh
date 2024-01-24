#!/bin/bash

# Dropbox URL
url="https://www.dropbox.com/scl/fi/6osc6bxsw33a7q71wov0y/weights.best.h5?rlkey=lr1a3b06u5e1zp33lt64276yw&dl=0"

# Convert Dropbox URL for direct download
direct_url="${url//www.dropbox.com/dl.dropboxusercontent.com}"
direct_url="${direct_url%?dl=0}?dl=1"

# Output file name
output="weights.best.h5"

# Download the file
wget -O "$output" "$direct_url"
