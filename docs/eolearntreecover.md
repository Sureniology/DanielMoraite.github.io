![# Welcome to my adventure](/images/Sat1.jpg)

# Tree Cover Prediction with Deep Learning 

## Keras, eo-learn, sentinel, satellite imagery, tensorflow



As you've seen in my previous article: [Satellite Imagery Analysis with Python. II](https://danielmoraite.github.io/docs/eo-learn.html) I had a look at a new library, `eo-learn`, which makes extraction of valuable information from satellite imagery easy. 

-------------------
Today will try one of the demos on Tree Cover Prediction that shows as well how easy is to use eo-learn for machine learning/ deep learning. Exactly training a U-net deep learning network to predict tree cover.

I have chosen an area of over 600sqm in UK(North-West of London). Geopedia/EU tree cover density has been used for gathering the ground-truth data.

___________________


### Setup 
    - install Sentinel Hub 
    - install eo-learn
    - install keras and tensorflow
(please find bellow, under resources, the links for the above)

#### Data Extraction

You'll find details of how to get your area of interest AOI coordinates in my previous: [Satellite Imagery Analysis with Python I](https://danielmoraite.github.io/docs/satellite1.html) post.
Make sure you save the coordinates in a file.geojson in your directory or if you have copied the github repo: ../eo-learn-master/example_data/.

 global image request parameters
 
    time_interval = ('2019-01-01', '2019-05-26')
    img_width = 240
    img_height = 256
    maxcc = 0.2

 get the AOI and split into bboxes
 
    crs = CRS.UTM_31N
    aoi = geopandas.read_file('../../example_data/europe.geojson')
    aoi = aoi.to_crs(crs={'init':CRS.ogc_string(crs)})
    aoi_shape = aoi.geometry.values.tolist()[-1]

    bbox_splitter = BBoxSplitter([aoi_shape], crs, (19, 10))

 set raster_value conversions for our Geopedia task
 see more about how to do this here:

    raster_value = {
        '0%': (0, [0, 0, 0, 0]),
        '10%': (1, [163, 235,  153, 255]),
        '30%': (2, [119, 195,  118, 255]),
        '50%': (3, [85, 160, 89, 255]),
        '70%': (4, [58, 130, 64, 255]),
        '90%': (5, [36, 103, 44, 255])
    }

    import matplotlib as mpl

    tree_cmap = mpl.colors.ListedColormap(['#F0F0F0', 
                                           '#A2EB9B', 
                                           '#77C277', 
                                           '#539F5B', 
                                           '#388141', 
                                           '#226528'])
    tree_cmap.set_over('white')
    tree_cmap.set_under('white')

    bounds = np.arange(-0.5, 6, 1).tolist()
    tree_norm = mpl.colors.BoundaryNorm(bounds, tree_cmap.N)

create a task for calculating a median pixel value

    class MedianPixel(EOTask):
        """
        The task returns a pixelwise median value from a time-series and stores the results in a 
        timeless data array.
        """
        def __init__(self, feature, feature_out):
            self.feature_type, self.feature_name = next(self._parse_features(feature)())
            self.feature_type_out, self.feature_name_out = next(self._parse_features(feature_out)())

        def execute(self, eopatch):
            eopatch.add_feature(self.feature_type_out, self.feature_name_out, 
                                np.median(eopatch[self.feature_type][self.feature_name], axis=0))
            return eopatch


initialize tasks
task to get S2 L2A images

    input_task = S2L2AWCSInput('TRUE-COLOR-S2-L2A', resx='10m', resy='10m', maxcc=0.2)
task to get ground-truth from Geopedia

    geopedia_data = AddGeopediaFeature((FeatureType.MASK_TIMELESS, 'TREE_COVER'), 
                                   layer='ttl2275', theme='QP', raster_value=raster_value)
task to compute median values

    get_median_pixel = MedianPixel((FeatureType.DATA, 'TRUE-COLOR-S2-L2A'), 
                               feature_out=(FeatureType.DATA_TIMELESS, 'MEDIAN_PIXEL'))
task to save to disk

    save = SaveToDisk(op.join('data', 'eopatch'), 
                  overwrite_permission=OverwritePermission.OVERWRITE_PATCH, 
                  compress_level=2)

 initialize workflow
 
    workflow = LinearWorkflow(input_task, geopedia_data, get_median_pixel, save)

 use a function to run this workflow on a single bbox
 
    def execute_workflow(index):
        bbox = bbox_splitter.bbox_list[index]
        info = bbox_splitter.info_list[index]

        patch_name = 'eopatch_{0}_row-{1}_col-{2}'.format(index, 
                                                          info['index_x'], 
                                                          info['index_y'])

        results = workflow.execute({input_task:{'bbox':bbox, 'time_interval':time_interval},
                                    save:{'eopatch_folder':patch_name}
                                   })
        return list(results.values())[-1]
        del results 

#### Test workflow on an example patch and display

    idx = 168
    example_patch = execute_workflow(idx)  
    example_patch


    EOPatch(
      data: {
        TRUE-COLOR-S2-L2A: numpy.ndarray(shape=(3, 427, 240, 3), dtype=float32)
      }
      mask: {
        IS_DATA: numpy.ndarray(shape=(3, 427, 240, 1), dtype=bool)
      }
      scalar: {}
      label: {}
      vector: {}
      data_timeless: {
        MEDIAN_PIXEL: numpy.ndarray(shape=(427, 240, 3), dtype=float32)
      }
      mask_timeless: {
        TREE_COVER: numpy.ndarray(shape=(427, 240, 1), dtype=uint8)
      }
      scalar_timeless: {}
      label_timeless: {}
      vector_timeless: {}
      meta_info: {
        maxcc: 0.2
        service_type: 'wcs'
        size_x: '10m'
        size_y: '10m'
        time_difference: datetime.timedelta(-1, 86399)
        time_interval: ('2019-01-01', '2019-05-26')
      }
      bbox: BBox(((247844.22638276426, 5741388.945588876), (250246.2123813057, 5745654.985149694)), crs=EPSG:32631)
      timestamp: [datetime.datetime(2019, 1, 17, 11, 16, 48), ..., datetime.datetime(2019, 2, 26, 11, 22, 1)], length=3
    )


    mp = example_patch.data_timeless['MEDIAN_PIXEL']
    plt.figure(figsize=(15,15))
    plt.imshow(2.5*mp)
    tc = example_patch.mask_timeless['TREE_COVER']
    plt.imshow(tc[...,0], vmin=0, vmax=5, alpha=.5, cmap=tree_cmap)
    plt.colorbar()

![# Welcome to my adventure](/images/TreeCover1.jpg)


### 2. Run workflow on all patches

 run over multiple bboxes
 
    subset_idx = len(bbox_splitter.bbox_list)
    x_train_raw = np.empty((subset_idx, img_height, img_width, 3))
    y_train_raw = np.empty((subset_idx, img_height, img_width, 1))
    pbar = tqdm(total=subset_idx)
    for idx in range(0, subset_idx):
        patch = execute_workflow(idx)
        x_train_raw[idx] = patch.data_timeless['MEDIAN_PIXEL'][20:276,0:240,:]
        y_train_raw[idx] = patch.mask_timeless['TREE_COVER'][20:276,0:240,:]
        pbar.update(1)

![# Welcome to my adventure](/images/TreeCover2.jpg)

### 3. Create training and validation data arrays

 data normalization and augmentation
 
    img_mean = np.mean(x_train_raw, axis=(0, 1, 2))
    img_std = np.std(x_train_raw, axis=(0, 1, 2))
    x_train_mean = x_train_raw - img_mean
    x_train = x_train_mean - img_std

    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180)

    y_train = to_categorical(y_train_raw, len(raster_value))


#### 4. Set up U-net model using Keras (tensorflow back-end)


 Model setup
 from https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2
 weight: weighted tensor(same shape with mask image)
 
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

 [check: weighted_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
 
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

    def weighted_dice_loss(y_true, y_pred, weight):
        smooth = 1.
        w, m1, m2 = weight * weight, y_true, y_pred
        intersection = (m1 * m2)
        score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
        loss = 1. - K.sum(score)
        return loss

    def weighted_bce_dice_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        # if we want to get same size of output, kernel size must be odd number
        averaged_mask = K.pool2d(
                y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
        border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
        weight = K.ones_like(averaged_mask)
        w0 = K.sum(weight)
        weight += border * 2
        w1 = K.sum(weight)
        weight *= (w0 / w1)
        loss = weighted_bce_loss(y_true, y_pred, weight) + \
        weighted_dice_loss(y_true, y_pred, weight)
        return loss

    def unet(input_size):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
                     kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6])
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', 
                     kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7])
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                     kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8])
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
                     kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9])
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                       kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(len(raster_value), 1, activation = 'softmax')(conv9)

        model = Model(inputs = inputs, outputs = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), 
                      loss = weighted_bce_dice_loss, 
                      metrics = ['accuracy'])

        return model

    model = unet(input_size=(256, 240, 3))

#### 5. Train the model

 Fit the model
    batch_size = 16
    model.fit_generator(
            train_gen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train),
            epochs=20,
            verbose=1)
    model.save(op.join('model.h5'))

#### 6. Validate model and show some results

 plot one example (image, label, prediction)
 
    idx = 4
    p = np.argmax(model.predict(np.array([x_train[idx]])), axis=3)
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(x_train_raw[idx])
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(y_train_raw[idx][:,:,0])
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(p[0])

![# Welcome to my adventure](/images/TreeCover3.jpg)

![# Welcome to my adventure](/images/TreeCover4.jpg)

![# Welcome to my adventure](/images/TreeCover5.jpg)


 plot one example (image, label, prediction)
 
    idx = 4
    p = np.argmax(model.predict(np.array([x_train[idx]])), axis=3)
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(x_train_raw[idx])
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(y_train_raw[idx][:,:,0])
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(p[0])
    
    
   show image confusion matrix
   
    predictions = np.argmax(model.predict(x_train), axis=3)
    cnf_matrix = confusion_matrix(y_train_raw.reshape(len(y_train_raw) * 256 * 256, 1), 
                                  predictions.reshape(len(predictions) * 256 * 256, 1))
    plot_confusion_matrix(cnf_matrix, raster_value.keys(), normalize=True)

![# Welcome to my adventure](/images/BR2NDVIderviedfromS2L1Cbands.png)

_________________________

Please find the entire code [here](https://github.com/DanielMoraite/DanielMoraite.github.io/blob/master/assets/forblogBR.ipynb)

Feel free to pick you own coordinates. You might play with the time frame as well..

_________________________
#### Resources:

  - [documentation for eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html)
  - [eo-learn github](https://github.com/sentinel-hub/eo-learn)
  - would recommend copying the eo-learn github repository on your drive
  - [sentinelhub create account](https://services.sentinel-hub.com/oauth/subscription?param_redirect_uri=https://apps.sentinel-hub.com/dashboard/oauthCallback.html&param_state=%2Fconfigurations&param_scope=SH&param_client_id=30cf1d69-af7e-4f3a-997d-0643d660a478&origin=)
  - [sentinelhub instance ID](https://www.sentinel-hub.com/faq/where-get-instance-id)
  - [sentinelhub install](https://pypi.org/project/sentinelhub/)
  - [keras](https://www.tensorflow.org/guide/keras#import_tfkeras)
  - [tensorflow](https://www.tensorflow.org/install)
  
_________________________
##### Notes:

    # SentinelHub API Key (instance id) stored as an env variable(if you want to call it from your notebook):        
        import os
        os.environ['INSTANCE_ID']='your instance ID'
        INSTANCE_ID = os.environ.get('INSTANCE_ID')



HAVE FUN!

-------------------------
_________________________
