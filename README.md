# Face-detection-and-Age-Gender-Classification-using-CNNs

Three pretrained CNNs are being used in this project. The model files can't be shared here due to the memory limits. Instead here is the link for the files.
**link for the models** ~ https://drive.google.com/drive/folders/1pGOFd_GwLfIY1YeambTaT7z6h9eXPf0G?usp=sharing

You can check the model architecture here ~ https://netron.app/

The frame **needs some preprocessing** before it can be fed to the networks.
**Mean value** used for preprocessing for the **face detection** = **(104.0, 177.0, 123.0)**.

**Mean value** used for preprocessing for the **age and gender classification** = **(78.4263377603, 87.7689143744, 114.895847746)**.

**No scaling is done**. So scale factor could be 1.0.

The frame array is preprocessed and fed to face detection network first.
Then the bounding boxes are framed by computing the co-ordinates from the predictions of face detection network.
The frame is cropped according to the co-ordinates.
Then the cropped ROI is fed to the age detection and gender detection models.
The results are obtained from the age and gender arrays using the predicted probablities returned by the networks.

Here OpenCV is used for preprocessing, live camera feeding and displaying the outcome.



