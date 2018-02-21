from __future__ import print_function
from data_preparation import load_data
from set_session import set_session
import resnet
import numpy as np
import keras
import os

output_dim = 102
im_dir = "../dataset/images/"
train_im_list = "../dataset/SUNAttributeDB/train.txt" 
test_im_list = "../dataset/SUNAttributeDB/test.txt"
checkpoint_dir = '../../Weights/ResNet-34/checkpoint-149-1.24.h5'

set_session(0)

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = load_data(im_dir, train_im_list, test_im_list)

model = resnet.ResnetBuilder.build_resnet_34((3, 224, 224), 102)
model.load_weights(checkpoint_dir)
# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.05, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

pred = model.predict(x_test)

# Evaluation
t_pos = np.zeros((output_dim,1), dtype='float32')
res = open('pred.txt', 'w')
for i in xrange(len(x_test)):
    for j in xrange(output_dim):
        if (pred[i][j] >= 0.5 and y_test[i][j] == 1) or (pred[i][j] < 0.5 and y_test[i][j] == 0):
            t_pos[j] += 1
    output = ' '.join(str(k) for k in pred[i])
    res.write(output)
    res.write('\n')
res.close()

average_precision = np.mean(t_pos/len(x_test), axis=0)

print(average_precision)
