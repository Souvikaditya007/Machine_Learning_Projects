import warnings
warnings.filterwarnings(action='ignore')

#Statndard scientfic Python Imports
import matplotlib.pyplot as plt

#Import datasets,classifiers and perfomannce metrics
from sklearn import datasets,svm

#The digits dataset
digits=datasets.load_digits()

print("digits: ",digits.keys())

print("Digits.target----- ",digits.target)

images_and_labels=list(zip(digits.images,digits.target))

print("len(images_and_labels)",len(images_and_labels))

for index,[image,label] in enumerate(images_and_labels[ : 5 ]):
    print("Index: ",index," image: \n",image,"Label: ",label)
    plt.subplot(2,5,index+1)  #Position numbering starts from 1
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training %i '% label)

#plt.show



#To apply a classifier on this data,we need to flattern the image
#to turn the data in a (samples,features) matrix:
n_samples=len(digits.images)   #n_samples=1797
print("n_samples: ",n_samples)

imageData=digits.images.reshape((n_samples,-1))
print("After Reeshaped: len(imageData[0]: ",len(imageData[0]))

#create a classifier: a support vector classifier
classifier=svm.SVC(gamma=0.001)

#We learn the digits on the first half of the digits
classifier.fit( imageData[   : n_samples//2],
                digits.target[  : n_samples//2]
              )

#Now predict the value of the digit on thr second half:
expectedY=digits.target[n_samples//2]

predictedY=classifier.predict(imageData[n_samples//2 :  ])

image_and_predictions=list(zip(digits.images[n_samples//2 : ],predictedY ))

for index,[image,prediction] in enumerate(image_and_predictions[ : 5 ]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('prediction: %i '% prediction)

print("Original Value: ",digits.target[n_samples//2 : (n_samples//2)+5 ])

plt.show()

#install pillow Library
from  scipy.misc import imread,imresize,bytescale

img=imread("Three.jpeg")            #Here add your own Image!
img=imresize(img, (8,8) )

classifier=svm.SVC(gamma=0.001)
classifier.fit( imageData[ : ],digits.target[ : ])

img=img.astype(digits.images.dtype)
img=bytescale(img,high=16.0,low=0)

print("img.shape: ",img.shape)
print("\n ",img)

x_testData=[]

for c in img:
    for r in c:
        x_testData.append((sum(r)/3.0))

print("x_testData: \n",x_testData)

print("len(x_testdata): ",len(x_testData))
print("Machine output: ",classifier.predict([x_testData]))

plt.show()
