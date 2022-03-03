#All radiographs (dicom files) were converted into numpyz files composed of x_train and y_train.
#x_train was a 224x224x3 pre-processed radiograph (image).

def img_transform(image_dicom):    
    image = image_dicom.pixel_array.astype(float)
    image_scaled = (np.maximum(image,0)/image.max())
    
    if(image_dicom.PhotometricInterpretation=="MONOCHROME1"):
        image_scaled = 1-image_scaled
        
    image_scale = np.uint8((np.maximum(image,0)/image.max())*255.0)
    image_scale = cv2.resize(image_scale, (img_width, img_height))
    
    image_resize = cv2.resize(image_scaled, (img_width, img_height))
    image_resize = (image_resize-np.mean(np.array(image_resize))) / np.std(np.array(image_resize))
    im_x = np.uint8((image_resize*0.229 + 0.485)*255.0)
    im_y = np.uint8((image_resize*0.224 + 0.456)*255.0)
    im_z = np.uint8((image_resize*0.225 + 0.406)*255.0)
    image_final = np.stack((im_x, im_y, im_z), axis = 2)
    
    return image_final

#y_train was a survival array (groundtruth) generated from survival time and censor information

y_train = make_surv_array(y_sur_train, censor_train, breaks)
y_val = make_surv_array(y_sur_val, censor_val, breaks)
y_test = make_surv_array(y_sur_test, censor_test, breaks)

#network: we used DenseNet169 pretrained with Imagenet

model_dense = applications.DenseNet169(include_top=False, 
                              weights='imagenet', 
                              input_shape=(img_width, img_height, 3))

x = model_dense.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(n_intervals, 
                    input_dim = 1, 
                    kernel_initializer='zeros', 
                    bias_initializer='zeros', 
                    activation = "sigmoid")(x)
model_final = Model(inputs = model_dense.input, 
                    outputs = predictions)

#train
from keras.optimizers import Adam

batch_size = 32
epochs = 100

checkpoint = ModelCheckpoint("/home/snuhrad/neuralnetworks/Dense.h5", 
                             monitor='val_loss', verbose=1, 
                             save_best_only=True, save_weights_only=False, 
                             mode='auto', period=1)

early_stopping = EarlyStopping(monitor='val_loss', 
                               min_delta=0, patience=20, 
                               verbose=1, mode='auto', restore_best_weights=True) 

model_final.compile(loss = surv_likelihood(n_intervals), 
                    optimizer = Adam(lr=0.0001, beta_1 = 0.9))

history = model_final.fit(x_train, y_train, 
                          batch_size=batch_size, 
                          epochs=epochs,
                          validation_data = (x_val,y_val),
                          callbacks=[early_stopping])
