#!/bin/bash

# URL of the file to download
FILE_ID="1sfVsdTxFI5gUsfSSxpbDpsZJlYNiLD4F"
FILE_NAME="data"

# Use wget to download the file from Google Drive
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O ${FILE_NAME}.zip

echo "Download completed: ${FILE_NAME}"

# Unzip the downloaded file into a folder
mkdir -p ${FILE_NAME}
unzip ${FILE_NAME}.zip -d ${FILE_NAME}
echo "Unzipped into folder: ${FILE_NAME}"

# Remove the zip file after extraction
rm ${FILE_NAME}.zip