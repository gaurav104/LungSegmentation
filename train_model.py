
from keras.preprocessing.image import ImageDataGenerator
from build_model import build_UNet2D_4L
from keras.optimizers import Adam

#import pandas as pd
#from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    img_size = (256,256)
    
    #Custom Loss
    import keras.backend as K
    def dice_coef(y_true, y_pred, smooth):
        #y_pred = K.cast(y_pred > thresh,dtype='float32')
        #y_true = K.cast(y_true, dtype='float32')
        #y_true_f = K.flatten(y_true)
        #y_pred_f = K.flatten(y_pred)
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
         ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    
    def binary_crossentropy(y_true, y_pred):
        loss = K.max(y_pred,0)-y_pred * y_true + K.log(1+K.exp((-1)*K.abs(y_pred)))
        return loss
    
    

    def dice_loss(smooth):
        def dice(y_true, y_pred):
            return 2*(1-dice_coef(y_true, y_pred, smooth))
        return dice
    
    def custom_loss(smooth):
        def bplusd(y_true, y_pred):
            return (1/50)*(binary_crossentropy(y_true, y_pred))+(1-dice_coef(y_true, y_pred, smooth))
        return bplusd
    
    def Bloss():
        def bcross(y_true, y_pred):
            return binary_crossentropy(y_true,y_pred)
        return bcross
            
        
    
    model_dice = dice_loss(1)
    
    optimizer = Adam(lr=0.001)

    # Build model
    inp_shape = (None,None,3)
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    #plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'model.{epoch:03d}.hdf5'
    print (model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    ''' 
    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)
    test_gen = ImageDataGenerator(rescale=1.)
    '''
    data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     rescale  = 1./255)
    image_datagen_train = ImageDataGenerator(featurewise_center=False,
                                       featurewise_std_normalization=False,
                                       rotation_range=10.,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2,
                                       rescale  = 1./255)
    mask_datagen_train = ImageDataGenerator(featurewise_center=False,
                                      featurewise_std_normalization=False,
                                      rotation_range=10.,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.2,
                                      rescale  = 1./255)
    image_datagen_test = ImageDataGenerator(rescale  = 1./255)
    mask_datagen_test = ImageDataGenerator(rescale  = 1./255)

# Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    #image_datagen.fit(images, augment=True, seed=seed)
    #mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator_train = image_datagen_train.flow_from_directory(
            'Image/Train/',
            class_mode=None,
            seed=seed,
            batch_size=1,
            target_size= img_size)

    mask_generator_train = mask_datagen_train.flow_from_directory(
            'Mask/Train/',
            class_mode=None,
            seed=seed,
            batch_size=1,
            target_size=img_size,
            color_mode='grayscale')
    
    image_generator_test = image_datagen_test.flow_from_directory(
            'Image/Test/',
            class_mode=None,
            seed=seed,
            batch_size=1,
            target_size= img_size)

    mask_generator_test = mask_datagen_test.flow_from_directory(
            'Mask/Test/',
            class_mode=None,
            seed=seed,
            batch_size=1,
            target_size=img_size,
            color_mode='grayscale')

# combine generators into one which yields image and masks
    train_generator = zip(image_generator_train, mask_generator_train)
    test_generator = zip(image_generator_test, mask_generator_test)


    #batch_size = 8
    UNet.fit_generator(train_generator,
                       steps_per_epoch=57,
                       epochs=20,
                       callbacks=[checkpointer],
                       validation_data = test_generator,
                       validation_steps = 20)
                                             