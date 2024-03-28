import os

# "test" for Validation data
# "train" for training data
path = "test"

for folders, subfolders, images in os.walk(path):
    for img in images:
        if folders.endswith("female"):
            with open((path + "_" + "data.txt"), "a") as f:
                f.write(img + "," + os.path.join(folders, img) + "," + "0\n")
        else:
            with open((path + "_" + "data.txt"), "a") as f:
                f.write(img + "," + os.path.join(folders, img) + "," + "1\n")
                