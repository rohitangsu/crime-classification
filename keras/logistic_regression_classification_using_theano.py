from scipy import ndimage, misc
import numpy as np
import os, re
import theano
import theano.tensor as T
rng = np.random

""" --------------------------------------------------------------------------

     Logistic Regression model for classification using theano
-------------------------------------------------------------------------- """


class_label = {'cat': 0, 'dog': 1}

def get_images(folderpath, N=10000):
    images = []
    labels = []
    counter = 0
    for root, dirnames, filenames in os.walk(folderpath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                # File name is of the form: dog.8131.jpg
                label = class_label[filename.split('.')[0]]
                labels.append(float(label))
                
                if counter % 1000 == 0:
                    print 'counter: ', counter , 'at', filepath
                counter += 1
 
                image = ndimage.imread(filepath, mode="RGB")
                image_resized = misc.imresize(image, (100,100))
                image_resized = image_resized.flatten()
                images.append(image_resized)
                
                if counter == N:
                	break
        if counter == N:
        	break

    return np.array(images), np.array(labels)
    
FOLDER_PATH = '/home/rohit/projects/crime-classification/keras/train'

images, labels = get_images(FOLDER_PATH, 20000)
print 'Images are fetched..'
import pdb
pdb.set_trace()
#images = images.astype(theano.config.floatX)
#labels = labels.astype(theano.config.floatX)

features = images.shape[1]

training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(features).astype(theano.config.floatX), name="w")
b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
x.tag.test_value = images
y.tag.test_value = labels

print 'Variables are initialized..'

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # Probability of having a one
prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # Cross-entropy
cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
gw,gb = T.grad(cost, [w,b])

print 'Constructing the symbolic expression graph..'
# Compile expressions to functions
train = theano.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates=[(w, w-0.01*gw), (b, b-0.01*gb)],
            name = "train")
predict = theano.function(inputs=[x], outputs=prediction,
            name = "predict")

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if theano used the cpu or the gpu')
    print(train.maker.fgraph.toposort())

for i in range(training_steps):
    pred, err = train(images, labels)


print 'Finished training the model..'
print("target values for D")
print(labels)

print("prediction on D")
print(predict(images))


