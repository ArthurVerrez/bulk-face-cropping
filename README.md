# Bulk Face Cropping
## Overview
Simple utility based on face recognition to crop out a given face out of multiple pictures.

The idea is that, if you need to make a dataset with someone's face cropped out properly, it can be very long to do it by hand. The goal is to automate this action.

This utility takes all of the images in a folder, recognizes the input face in the images, crops the pictures and saves them in another folder. It works even when there are several people in the pictures.

This utility is almost entirely based on the [face-recognition package](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

## Motivation
Leveraging [Google Photos' face grouping feature](https://support.google.com/photos/answer/6128838?hl=en&co=GENIE.Platform%3DAndroid), to bunk download all of the pictures of a given person, and training a GAN to generate realistic faces.
It could now be used today to create a database to fine-tune [Stable Diffusion with Dreambooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) for example.

## Example

For example, it will take this image:
![Meghan Markle and prince Harry](demo_images/input/harry-meghan.jpg)

And using the encoding of this image:
![Meghan Markle's face](demo_images/ref_image.jpg)

It will output this one:

![Meghan Markle's face cropped](demo_images/output/harry-meghan-cropped.jpg)

# Installation
## Requirements
This utility was made for Python 3.10 and Linux (or WSL-2 on Windows) and was not tested on other versions or systems.

```bash
git clone https://github.com/ArthurVerrez/bulk-face-cropping.git
cd bulk-face-cropping
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*The installation of face_recognition might take a while because it has to build dlib from source.*

## Usage
```bash
python3 extract_faces.py -i <input_folder> -o <output_folder> -f <face_refence_image>
```
*You can also add -v for verbose mode.*

*The **<face_reference_image>** file must only contain one face.*

## Example usage
```bash
python3 extract_faces.py -i demo_images/input -o demo_images/output -f demo_images/ref_image.jpg
```

Enjoy!