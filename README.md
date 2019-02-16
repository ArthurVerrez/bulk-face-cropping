# Automatic Face Cropping
Small algorithm based on face recognition to crop out a given face out of multiple pictures.

The idea is that, if you need to make a dataset with someone's face cropped out properly, it is very long to do it by hand over the pictures you have of someone. The goal is to automatize this action.

This small algorithms takes all of the images in a folder, recognizes the wanted face in the image, crop the picture and save it in another folder. It works even when there are several pictures in the picture.

This algorithm is entirely based on the [face_recognition package](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

For example, it will take this image:
![Meghan Markle and prince Harry](/harry-meghan.jpg)

And output this one:

![Meghan Markle's face cropped](/0.jpg)
