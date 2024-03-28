# Male-Female-Classifier
In this project, I will explore the Male and Female Dataset scrapped from multiple e-commerce websites and build a model that can predict their corresponding gender.<br />
I used Tensorboard for live results of training and validation.
### You need to scrap Images for dataset by using scrapper.py
### Note: It will work only on models images(fashion models) because this model trained on images scrapped from e-commerce websites.
## INSTRUCTIONS
### This project requires the following libraries :
•	[Os](https://python.readthedocs.io/en/stable/library/os.html)<br />
•	[Torch(Pytorch)](https://pytorch.org/docs/stable/index.html)<br />
•	[Numpy](https://numpy.org/)<br />
•	[Cv2(OpenCV)](https://docs.opencv.org/4.x/)<br />
• [Requests](https://requests.readthedocs.io/en/latest/)<br />
•	[Selenium](https://selenium-python.readthedocs.io/)<br />
•	[Matplotlib](https://matplotlib.org/stable/index.html)<br />
• [torch.utils.tensorboard](https://www.tensorflow.org/tensorboard/get_started)<br />

### Please ensure you have installed the following libraries mentioned above before continuing.<br />

## HOW TO CHECK PREDICTION

To check predictions Extract all files in one folder.<br /><br />
Run CMD in folder directory and type:
```
python prediction.py --img_root "Image path"
```
Example:
```
python prediction.py --img_root "C:/user/data/img.jpg"
```
An Image will be open in another windows with prediction printed on it.<br /><br />

## Results

| Training Result  | Validation Result |
| ------------- | ------------- |
| <img src="https://user-images.githubusercontent.com/97089717/208094741-93d414f1-eb35-4b09-b5bb-6f29fef43967.svg" width="500" height="500">  | <img src="https://user-images.githubusercontent.com/97089717/208095432-7c3e0f45-77be-45d0-b32e-9b7aa3f72af3.svg" width="500" height="500">  |
