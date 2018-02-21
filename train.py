from __future__ import print_function
from data_preparation import load_data
from set_session import set_session
import resnet
import keras
import os

batch_size = 64
epochs = 150
im_dir = "../dataset/images/"
train_im_list = "../dataset/SUNAttributeDB/train.txt" 
test_im_list = "../dataset/SUNAttributeDB/test.txt"
set_session(0)
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = load_data(im_dir, train_im_list, test_im_list)

model = resnet.ResnetBuilder.build_resnet_34((3, 224, 224), 102)

# initiate SGD optimizer
opt = keras.optimizers.SGD(lr=0.05, decay=1e-6)

# Let's train the model using SGD
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Not using data augmentation.')
checkpoint = keras.callbacks.ModelCheckpoint(filepath='./checkpoint/checkpoint-{epoch:02d}-{val_loss:.2f}.h5')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpoint])

