#import Vgg_network as vg
#import matplotlib.pyplot as plt
#from PIL import Image
#
#
#facemodel,description = vg.load_Save()
im = Image.open('ak.png')
im = im.resize((224,224))
f=vg.features(facemodel, im, transform=False)
print(f)
print(f.shape)
#vg.pred(facemodel, im, description,transform=False)
#vg.pred(facemodel, im,description,transform=True)