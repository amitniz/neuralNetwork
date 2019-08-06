from PIL import Image
import os

size = (28,28)
dir =  os.listdir("data")
j=0

for k in os.listdir("./photos"):
    if k.endswith('.jpg'):
        j+=1

for file in dir:
    if file.endswith('.jpg'):
        print(file)
        i = Image.open("data/{}".format(file))
        i.thumbnail(size)
        i = i.resize(size)
        i.save('photos/{}.jpg'.format(j))
        os.remove("data/{}".format(file))


