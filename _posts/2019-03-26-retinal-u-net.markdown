---
layout: post
title:  Retinal Vasculature Segmentation with a U-Net Architecture
date:   2019-03-26 00:10:45
categories: Self-Initiated Project
excerpt_separator: <!--more-->
---
The structures exhibited by the retinal vasculature infer critical information about a wide range of retinal pathologies such as Prematurity (RoP), Diabetic Retinopathy(DR), Glaucoma, hypertension, and Age-related Macular Degeneration(AMD). These pathologies are amongst the leading causes of blindness. Accurate segmentation of retinal vasculature is important for various ophthalmologic diagnostic and therapeutic procedures.
<!--more-->


A lot of research have focused on developing automated and accurate techniques for retinal vessel segmentation over the past few decades. With the rise of Machine Learning, Deep Learning and Computer Vision in the recent years, researchers have found ways to apply these technologies to provide solutions for the problems present in Medicine, Biology and Healthcare.


U-Net is an interesting deep learning network architecture amongst these technologies. It was developed by O. Ronneberger, P. Fischer, and T. Brox in 2015 and can be categorized as a fully Convolution Neural Network (CNN) for Biomedical Image Segmentation. The authors of the U-Net paper wrote the following.

> *…In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization…*

U-Net does not need a huge amount of training data and that makes it ideal for Biomedical image analysis because of the relative scarcity of images in the field of Biomedicine. In this article, we will discuss how to write up a simple U-Net architecture to solve the retinal vessel segmentation problem and how to evaluate the performance of the algorithm.

<div style="text-align: center"><img src="{{site.url}}/images/fundus.jpeg" width="300" height="300" /></div>

I have used the “DRIVE: Digital Retinal Images for Vessel Extraction” dataset for training the network. In the dataset, there are two folders, namely ‘training’ and ‘test’. The ‘training’ folder contains 20 retinal images and their vessel masks. 17 images and their vessel masks from the ‘training’ folder were taken as the training set. The remaining 3 images and their vessel masks were taken as the validating set. The test folder contains 20 images and two types of vessel masks (1st_manual and 2nd_manual). The 1st_manual vessel masks were taken as the golden standard so that the human-annotations (2nd_manual) could be compared against the gold standard when evaluating the performance. The 20 images and their vessel masks (1st_manual) were taken as the testing data. The retinal images are 3 channel images (RGB) while their vessel masks are binary images. The original images from DRIVE are of the size, 565 × 584. They were resized to 512 × 512 before saving the training, validating and testing sets in a ‘.hdf5’ file.

The image below illustrates the U-Net architecture that we would be considering.

<div style="text-align: center"><img src="{{site.url}}/images/unet.png" width="300" height="300" /></div>

The following gist contains the U-Net architecture that we can use for training our model. The architecture is written in keras.

{% highlight ruby %} 

class UNET(object):
    def __init__(self, img_rows=256, img_cols=256, channel=3, n_filters=16, dropout=0.1, batchnorm=True ):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.model = None
        print('Input shape : (%i, %i, %i)' % (self.img_rows, self.img_cols, self.channel))

    def conv2d_block(self, input_tensor, filters, kernel_size=3):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def unet(self):

        input_img = Input((self.img_rows, self.img_cols, self.channel))

        # Contracting Path
        c1 = self.conv2d_block(input_img, filters=self.n_filters * 1, kernel_size=3)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout)(p1)

        c2 = self.conv2d_block(p1, filters=self.n_filters * 2, kernel_size=3)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)

        c3 = self.conv2d_block(p2,filters=self.n_filters * 4, kernel_size=3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)

        c4 = self.conv2d_block(p3, filters=self.n_filters * 8, kernel_size=3)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)

        c5 = self.conv2d_block(p4, filters=self.n_filters * 16, kernel_size=3)

        # Expansive Path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = self.conv2d_block(u6, filters=self.n_filters * 8, kernel_size=3)

        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = self.conv2d_block(u7, filters=self.n_filters * 4, kernel_size=3)

        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = self.conv2d_block(u8, filters=self.n_filters * 2, kernel_size=3)

        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(self.dropout)(u9)
        c9 = self.conv2d_block(u9, filters=self.n_filters * 1, kernel_size=3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        self.model = Model(inputs=[input_img], outputs=[outputs])

        # self.model.summary()
        # plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return self.model

{% endhighlight %}

Since our training set is fairly small, it is helpful to use data augmentation techniques to enhance the size of our training data. For this, we can use the ImageDataGenerator class of keras. It enables you to configure the image preparation and augmentation. The great thing about this class is its ability to create augmented data during the model fitting process itself. The data generator is actually an iterator which returns batches of augmented images when they are requested by the model fitting algorithm.

To prepare our data for the training, we have to first rescale them within the interval [0, 1]. When augmenting our data, we can use random rotations. The degree range of these random rotation can be specified in the datagenerator. This would make our model rotation invariant as the model would be seeing images of different orientations. Also, we can use horizontal and vertical random shifting as an augmentation technique. By training our model on images with different vertical and/or horizontal shifts, we can make our model translational invariant. Zooming is another augmentation technique we can use. That would make our model scale invariant. We can configure the above image data preparation and augmentation techniques as below.

{% highlight ruby %} 

datagen_args = dict(
             rescale=1./255,
             rotation_range=90,
             width_shift_range=0.1,
             height_shift_range=0.1,
             zoom_range=0.2
)

{% endhighlight %}

During the data preparation and augmentation, we have to make sure that the masks are getting the same changes that we are applying to the images. The following function would take care of that.

{% highlight ruby %} 

def get_generator(self, images, masks):
    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    seed = 1

    image_generator = image_datagen.flow(images, seed=seed)
    mask_generator = mask_datagen.flow(masks, seed=seed)

    return zip(image_generator, mask_generator)

{% endhighlight %}

Now, we can define the training routine of our model. We will use the Adam Optimizer at a learning rate of 0.0001. The loss function would be binary cross entropy since we are dealing with a pixel-wise labelling problem. (vessel regions = 1, non-vessel regions = 0). We would train the model for 50 epochs while having 200 steps per epoch and 32 as our batch size. That way, the model would be seeing 32 × 200 = 6400 images at each epoch thanks to image augmentation configuration that we defined earlier. We would save our model weights to a ‘.hdf5’ file whenever the loss is improved at the end of a given epoch. Also we would be implementing an early-stopper with a patience (number of epochs with no improvement after which training will be stopped) of 10 epochs.

{% highlight ruby %} 

compile_args = dict(
            optimizer=Adam(lr=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
)
earlystopper = EarlyStopping(
            patience=10,
            verbose=1
)
model_checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='loss',
            verbose=1,
            save_best_only=True
)
model.compile(**self.compile_args)
train_generator = self.get_generator(x_train, y_train)
val_generator = self.get_generator(x_val, y_val)
model.fit_generator(
            train_generator,
            epochs=50,
            steps_per_epoch=200,
            verbose=1,
            callbacks=[model_checkpoint, earlystopper],
            validation_data=val_generator,
            validation_steps=10
)

{% endhighlight %}

After the training is complete, we can evaluate our model. To evaluate the model, we can use performance metrics like F1-score, accuracy score, receiver-operator curve (ROC) AUC and precision-recall (PR) curve AUC. Plotting PR and ROC curves can provide a good insight about the model performance.

We can also plot a selected retinal image (from the test set) , its vessel mask created by the human annotator and its vessel mask predicted by our U-Net model.

<div style="text-align: center"><img src="{{site.url}}/images/retinal_results.png" width="500" height="200" /></div>

That winds up this article. Let me know your questions in the comments. Happy U-Netting!

References :
https://machinelearningmastery.com/image-augmentation-deep-learning-keras/