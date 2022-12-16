import cv2
import torch
import male_vs_female
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--img_root", type=str, default="image.jpg", help="Image Path")

opt = parser.parse_args()
image_path = opt.img_root
classes = male_vs_female.classes
transform = male_vs_female.test_transform

image = cv2.imread(image_path)
image = transform(image) 

model = male_vs_female.Model()
model.load_state_dict(torch.load('male_female_classifier.pt', map_location=torch.device('cpu')))
model.eval()

def prediction():
    with torch.no_grad():
        pred = model(image)
        predicted = torch.max(pred, 1)[1]
        
    img = male_vs_female.tensor_to_image(image)
    predicted_img = cv2.putText(img, ("Prediction: " + classes[predicted]), (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)
    predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
    plt.imshow(predicted_img)
    plt.show()
    
if __name__ == '__main__':
    prediction()
