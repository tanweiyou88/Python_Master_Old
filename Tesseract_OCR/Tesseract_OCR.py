"""
Project: OCR using Tesseract
Aim:
1) Extract texts from an image at a time
2) Extract texts from multiple images of the same file type in the same directory at a time
Youtube tutorial repository: https://github.com/JayMartMedia/python-tesseract
Tesseract Official Repository: https://tesseract-ocr.github.io/tessdoc/Installation.html
Tesseract Downloader Official Repository: https://github.com/UB-Mannheim/tesseract/wiki
Complete step of this project:
1) Extract texts from an image at a time
a) Specify the location and name of the image whose texts will be extracted through OCR
b) Use functions to open the image, then run pytesseract function to extract the texts from the image
c) print the texts extracted from the image
2) Extract texts from multiple images of the same file type in the same directory at a time
a) Specify the directory of the folder which containing the images whose texts will be extracted through OCR
b) Use glob('*.[file extension]') to find all files in the specified directory with the specified file extension
c) Loop over each file found
d) Use functions to open the image, then run pytesseract function to extract the texts from the image
e) print the texts extracted from the image
"""

import os
import pytesseract
from PIL import Image

# ## Extract texts from an image at a time
# file_path = os.path.join('D:\Python_Master\Tesseract_OCR','data','OIP.jpeg')
# img1 = Image.open(file_path)
# print(pytesseract.image_to_string(img1, lang = 'eng'))

## Extract texts from multiple images of the same file type in the same directory at a time
from pathlib import Path

directory = os.path.join('D:\Python_Master\Tesseract_OCR','data')
files = Path(directory).glob('*.jpg') # * means find all files in the specified directory with the specified file extension (.jpg in this case). glob() is used to find, locate, and search for files present in a system.
for file in files: # loop over each file
        print(file) # print each file name
        print(pytesseract.image_to_string(Image.open(file), lang = 'eng')) # run tesseract for each file
        print('\n-----------------\n') # print visual separator to separate the texts extracted from different files

