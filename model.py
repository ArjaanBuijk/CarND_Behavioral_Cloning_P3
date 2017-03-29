# Initial framework created following steps outlined here: https://goo.gl/siYhbo

import csv
import cv2
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from skimage import exposure
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os.path

myCLIP_LIMIT = 0.2 # for clahe transform
STEERING_ANGLE_CORRECTION = 0.2 
FILES_TOP_DIRS = ['./data/data_track_2_left','./data/data_track_2_right',
                  './data/data_track_1_left','./data/data_track_1_right',
                  ]

def read_log_files(top_dirs):
    lines = []
    for FILES_TOP_DIR in top_dirs:
        log_file_name = FILES_TOP_DIR+'/driving_log.csv'
        with open(log_file_name) as csvfile:
            reader = csv.reader(csvfile)
            skipped_header = False
            for line in reader:
                if skipped_header:
                    line.insert(0,FILES_TOP_DIR) # insert top directory as first token
                    lines.append(line) 
                else:
                    skipped_header=True    # skip header line
    return lines

def plot_image_and_augmented_image(image, image_augm, filename=None,
                                   txt='---'):
   
    nrows       = 2
    ncols       = 3            
    axes_width  = 6            
    axes_height = 1            
    width       = ncols * axes_width    
    height      = nrows * axes_height  
    fontsize    = 15 
    fig, axes   = plt.subplots(nrows, ncols, figsize = (width, height) )
          
    # turn off:
    #  - all tick marks and tick labels
    #  - frame of each axes
    for row in range(nrows):
        for ncol in range(ncols):
            axes[row,ncol].xaxis.set_visible(False)
            axes[row,ncol].yaxis.set_visible(False)
            axes[row,ncol].set_frame_on(False)
          
          
    # Header of columns
    row = 0
    axes[row, 0].text(0.0, 0.25, 
                      'Augmentation Operation',
                      fontsize=fontsize)
    axes[row, 1].text(0.4, 0.25, 
                      'Image',
                      fontsize=fontsize)
    axes[row, 2].text(0.4, 0.25, 
                      'Augmented Image',
                      fontsize=fontsize)
              
    
    row=1
    axes[row, 0].text(0.0, 0.25, 
                      (txt),
                    fontsize=fontsize)    
    
    if image.ndim == 3 and image.shape[2] == 3:
        axes[row,1].imshow(image)
    else:
        axes[row,1].imshow(image.squeeze(), cmap='gray')
        
    if image_augm.ndim == 3 and image_augm.shape[2] == 3:
        axes[row,2].imshow(image_augm)
    else:
        axes[row,2].imshow(image_augm.squeeze(), cmap='gray')


    if filename == None:      
        plt.show()  
    else:  
        # When running python directly, not in Jupyter notebook, it is better to
        # write it to a file & view it in an image viewer
        fig.savefig(filename)
        print ('Written the file: '+ filename)
          
    plt.close(fig)

def save_augmented_images_to_disk(save_to_dir, save_prefix, save_format, X_batch, y_batch):
    # TODO: this one does not correctly plot the clahed images...
    for i in range(len(X_batch)):
        img = array_to_img(X_batch[i], scale=False) # create a PIL image
        fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=save_prefix,
                                                          index=i,
                                                          hash=np.random.randint(1e4),
                                                          format=save_format)
        img.save(os.path.join(save_to_dir, fname))    

def rgb_to_grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    
    input: image"""
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # shape (160, 320)
    return x.reshape(x.shape[0], x.shape[1], 1) # shape (160, 320, 1) required in rest
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   

    
def apply_clahe(img, clip_limit=0.01):
    """Applies a Contrast Limited Adaptive Histogram Equalization (CLAHE)
    for description:
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    """
    x = exposure.equalize_adapthist(img.squeeze(), clip_limit=clip_limit)
    return x.reshape(x.shape[0], x.shape[1], 1) # shape (160, 320, 1) required in rest


def prep_for_yield(save_to_dir, save_prefix, save_format, images, angles):
    # yield the batch as numpy arrays
    X_batch = np.array(images)
    y_batch = np.array(angles)
    # write it to disk for debug purposes if requested
    if save_prefix:
        save_augmented_images_to_disk(save_to_dir, save_prefix, save_format,
                                      X_batch, y_batch)
    # return it, after shuffling, and calling function will yield it
    return sklearn.utils.shuffle(X_batch, y_batch)
    
def generator(lines, batch_size=32,
              flip_horizontal=False,
              grayscale=False,
              clahe=False,
              save_to_dir=None,
              save_prefix=None,
              save_format='jpeg'):
    '''
    Custom generator for behavior cloning exercise that yields batches for both 
    training & labels.
    
    inputs:
    - lines: the content of the log file created during data acquisition
    - batch_size: the size of batches to be yielded
    - flip_horizontal: True -> each image (center, left, right) will be also flipped
    - grayscale: True -> each image is grayscaled as well
    - clahe: True -> a clahe transform is applied to each image
    - prefix: path_and_prefix for file of final augmented images. Handy for debugging.

    Notes on difference with Keras ImageDataGenerator:
    - Keras ImagaDataGenerator only processes the images, not the labels. This does
      not allow to flip the steering angle during horizontal flip of the image.
    - Keras ImageDataGenerator, when flip_horizontal=True, either returns the image 
      itself, or its flipped image. Not both, like this generator. Thus, the data is not
      actually increased.
    '''
    assert(len(lines)>0)
    
    num_samples = len(lines) * 6  # center, left, right, and each one flipped
    
    while True: # Loop forever so the generator never terminates
        images = []
        angles = []    
        count = 0   
        total_count = 0
        fnames = ['']*3
        for line in lines:
            FILES_TOP_DIR = line[0].strip()
            fnames[0]  = FILES_TOP_DIR + '/' + line[1].strip() # center
            
            if line[2].strip() == 'EMPTY':
                fnames[1] = 'EMPTY'
            else:
                fnames[1]  = FILES_TOP_DIR + '/' + line[2].strip() # left
                
            if line[3].strip() == 'EMPTY':
                fnames[2] = 'EMPTY'
            else:
                fnames[2]  = FILES_TOP_DIR + '/' + line[3].strip() # left                

            steering  = float(line[4])

            for f_index in range(3):
                f = fnames[f_index]
                
                if f_index == 0:    # center
                    angle = steering
                elif f == 'EMPTY':
                    # no left or right image provided. Just duplicate the center image.
                    angle = steering 
                    f = fnames[0]
                elif f_index == 1:  # left
                    angle = steering + STEERING_ANGLE_CORRECTION
                else:               # right
                    angle = steering - STEERING_ANGLE_CORRECTION
                    
                if os.path.isfile(f):
                    image = cv2.imread(f)
                    
                    # apply image manipulations
                    if grayscale:
                        image = rgb_to_grayscale(image)
                        
                    if clahe:
                        image = apply_clahe(image, clip_limit=myCLIP_LIMIT)
                        
                    
                    images.append(image)
                    angles.append(angle)
                    count += 1
                    total_count += 1
                    if count == batch_size or total_count == num_samples:
                        yield prep_for_yield(save_to_dir, save_prefix, save_format, 
                                             images, angles)
                        
                        # reset for next batch
                        images, angles, count = [], [], 0
                        
                    if flip_horizontal:          
                        # augment image by horizontal flip
                        x = cv2.flip(image,1) # returns (160, 320)
                        image_flipped = x.reshape(x.shape[0], x.shape[1], 1) # (160, 320, 1)
                            
                        images.append(image_flipped)
                        angles.append(angle*-1.0)
                        count += 1 
                        total_count += 1
                        if count == batch_size or total_count == num_samples:
                            yield prep_for_yield(save_to_dir, save_prefix, save_format, 
                                                 images, angles)
                            
                            # reset for next batch
                            images, angles, count = [], [], 0
                        
                else:
                    msg = 'Image file does not exist: '+str(f)
                    raise Exception(msg)

if __name__ == '__main__':
    ## ======================================================================================
    ## Try out the data generator --> This is just for debugging purposes
    
    #lines_trial = read_log_files(['data/data_to_test_shadow_contrast/'])
    
    #trial_generator  = generator(lines_trial, batch_size=1,
                                #flip_horizontal=True,
                                #grayscale=True,
                                #clahe=True,
                                #save_to_dir='data/data_to_test_shadow_contrast/preview',
                                #save_prefix='test',
                                #save_format='jpeg')
    
    ## generate a few augmented images using the trial_generator, which
    ## will be written to the preview folder.
    #i = 0
    #for batch in trial_generator:
        #i += 1
        #if i > 4: 
            #break  # otherwise the generator would loop indefinitely
    
    # ======================================================================================
    print ('Train & Validate the network...')
    
    # 
    # read log files of labeled training images
    lines = read_log_files(FILES_TOP_DIRS)
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    #
    # Using custom generator
    #
    train_generator      = generator(train_samples, batch_size=32,
                                     flip_horizontal=True,
                                     grayscale=True,
                                     clahe=True,
                                     save_to_dir=None,
                                     save_prefix=None,
                                     save_format='jpeg')
    
    validation_generator = generator(validation_samples, batch_size=32,
                                     flip_horizontal=False, # do NOT generate new images during validation
                                     grayscale=True,
                                     clahe=True,
                                     save_to_dir=None,
                                     save_prefix=None,
                                     save_format='jpeg')
                                                                        
    
    # define Keras network
    # it is a regression network, not a classification network !
    # -> Just predict steering angle, no probability/softmax
    # -> For loss function, use MSE (Mean Squared Error), not cross-entropy
    
    model = Sequential()
    #
    # Pre-process the data as part of Keras model
    #
    # crop:
    # 70 rows pixels from the top of the image
    # 25 rows pixels from the bottom of the image
    # 0 columns of pixels from the left of the image
    # 0 columns of pixels from the right of the image
    #rgb 
    #model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3), name='layer_crop'))
    #grayscale
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,1), name='layer_crop'))
    #
    # normalize and mean-center the data
    #without clahe:
    #model.add(Lambda(lambda x: x / 255.0 - 0.5, name='layer_normalize'))
    #with clahe
    model.add(Lambda(lambda x: x / 1.0 - 0.5, name='layer_normalize'))
    
    # Nvidia CNN
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    
    # Diagnostics of Preprocessing
    # see: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # get first & second image (center and flipped)
    # note: next() returns a list [---,], where --- is the numpy array of images.
    i = 0
    for batch in train_generator:
        i += 1
        
        X_batch         = batch[0]    
        
        ## get cropped immage
        layer_name = 'layer_crop'
        layer_model = Model(input=model.input,
                            output=model.get_layer(layer_name).output)
        layer_images = layer_model.predict(X_batch)
        plot_image_and_augmented_image(X_batch[0], 
                                       layer_images[0],            # plot the first image, cropped
                                       filename='first_image_cropped',
                                       txt='Cropped input image')    
        
        ## get normalized immage
        layer_name = 'layer_normalize'
        layer_model = Model(input=model.input,
                            output=model.get_layer(layer_name).output)
        layer_images = layer_model.predict(X_batch)
        plot_image_and_augmented_image(X_batch[0], 
                                       layer_images[0],            # plot the first image, cropped
                                       filename='first_image_normalized',
                                       txt='Normalized input image')        
    
        if i > 0: 
            break  # otherwise the generator would loop indefinitely
        
        
        
    # train & test the model
    model.fit_generator( train_generator, 
                         samples_per_epoch=len(train_samples*6), # center, left, right, flipped*3
                         validation_data=validation_generator,
                         nb_val_samples=len(validation_samples),
                         nb_epoch=10 )
    
    
    
    model.save('model.h5')
