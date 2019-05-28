![# Welcome to my adventure](/images/Sat1.jpg)

# Satellite Imagery Analysis with Python. II


## eo-learn

I guess I have a thing for keeping up with the newest libraries and repositiories for manipumating data from more and more data intensive sources. 
Of course, I wouldn't have had introduce you to them unless they aide and facilitate the application of machine learning to solve problems and find innovative solutions:
  - efficiently handle big data      - handle different data sources and formats     - complete workflows
  - handle spatio-temporal data      - prototype, build and automate                 - deal with lack of training labels
  - create and validate machine learning applications                                - deal with missing data

Today, I would like to introduce you to an upcoming amazing library: eo-learn. 

`eo-learn` not only makes extraction of valuable information from satellite imagery easy, is has as well a very high goal: to democratise Earth observation big data.
It started as a response to the availability of open Earth observation (EO) data through the Copernicus and Landsat programs. 
And to quote:
> is a collection of open source Python packages that have been developed to seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner.
> eo-learn library acts as a bridge between Earth observation/Remote sensing field and Python ecosystem for data science and machine learning.

-------------------
In this post will have a look at the basics of using eo-learn and give it a go to downloading, saving and visualizing EO imaging data(RGB, NDVI, Scene Classification masks, digital elevation).


### Setup 
   - install Sentinel Hub 
   - install eo-learn
(please find bellow, under resources, the links for the above)

#### Data Extraction

You'll find details of how to get your area of interest `AOI` coordinates in my previous: [Satellite Imagery Analysis with Python I](https://danielmoraite.github.io/docs/satellite1.html) post. Just make sure you select from the menu: `meta`, respectively `add bboxes`. 

Define ROI BBOX and time interval  
I have picked an area around my home town, a beautiful agricultural area around Danube River. 

    roi_bbox = BBox(bbox=[27.67, 44.97, 28.03, 45.26], crs=CRS.WGS84)
    time_interval = ('2019-04-01', '2019-05-01')

Request different types of layers and data sources to an eopatch

      layer = 'BANDS-S2-L1C'
      input_task = S2L1CWCSInput(layer=layer, 
                                 resx='20m', resy='20m', 
                                 maxcc=.3, time_difference=datetime.timedelta(hours=2))
      add_ndvi = S2L1CWCSInput(layer='NDVI')
      add_dem = DEMWCSInput(layer='DEM')
      add_l2a = S2L2AWCSInput(layer='BANDS-S2-L2A')
      add_sen2cor = AddSen2CorClassificationFeature('SCL', layer='BANDS-S2-L2A')
      save = SaveToDisk('io_example', overwrite_permission=2, compress_level=1)
      
Sentinel2 L1C and L2A bands are requested (all of them 12-13bands) at 20m resolution, maxcc= max cloud coverage(you can play with it in between 0.8 and 0.05), and NDVI, DEM digital elevation model. 

Run workflow       
        
      workflow = LinearWorkflow(input_task, add_ndvi, add_l2a, add_sen2cor, add_dem, save)
      result = workflow.execute({input_task: {'bbox': roi_bbox, 'time_interval': time_interval},
                           save: {'eopatch_folder': 'eopatch'}})
        
 Check contents of eopatch
 
        eopatch = result[save]
        eopatch
        
      EOPatch(
        data: {
          BANDS-S2-L1C: numpy.ndarray(shape=(5, 1613, 1413, 13), dtype=float32)
          BANDS-S2-L2A: numpy.ndarray(shape=(5, 1613, 1413, 12), dtype=float32)
          NDVI: numpy.ndarray(shape=(5, 1613, 1413, 1), dtype=float32)
        }
        mask: {
          IS_DATA: numpy.ndarray(shape=(5, 1613, 1413, 1), dtype=bool)
          SCL: numpy.ndarray(shape=(5, 1613, 1413, 1), dtype=int32)
        }
        scalar: {}
        label: {}
        vector: {}
        data_timeless: {
          DEM: numpy.ndarray(shape=(1613, 1413, 1), dtype=float32)
        }
        mask_timeless: {}
        scalar_timeless: {}
        label_timeless: {}
        vector_timeless: {}
        meta_info: {
          maxcc: 0.3
          service_type: 'wcs'
          size_x: '20m'
          size_y: '20m'
          time_difference: datetime.timedelta(0, 7200)
          time_interval: ('2019-04-01', '2019-05-01')
        }
        bbox: BBox(((27.67, 44.97), (28.03, 45.26)), crs=EPSG:4326)
        timestamp: [datetime.datetime(2019, 4, 1, 9, 18, 16), ..., datetime.datetime(2019, 4, 28, 9, 8, 8)], length=5
      )
        
### Plot results

#### S2 L1C RGB bands

    plt.figure(figsize=(10,10))
    plt.imshow(eopatch.data['BANDS-S2-L1C'][3][..., [3,2,1]] * 2.5, vmin=0, vmax=1);
![# Welcome to my adventure](/images/BR1S2L1CRGBbands.png)

#### NDVI dervied from S2 L1C bands

    plt.figure(figsize=(10,10))
    plt.imshow(eopatch.data['NDVI'][3].squeeze());
![# Welcome to my adventure](/images/BR2NDVIderviedfromS2L1Cbands.png)

##### S2 L2A RGB bands

    plt.figure(figsize=(10,10))
    plt.imshow(eopatch.data['BANDS-S2-L2A'][3][...,[3,2,1]] * 2.5, vmin=0, vmax=1);
![# Welcome to my adventure](/images/BR3S2L2ARGBbands.png)

#### Sen2cor scene classification mask

    plt.figure(figsize=(10,10))
    plt.imshow(eopatch.mask['SCL'][3].squeeze());
![# Welcome to my adventure](/images/BR4Sen2corsceneclassificationmask.png)    

#### Mapzen Digital Elevation Model

    plt.figure(figsize=(10, 10))
    plt.imshow(eopatch.data_timeless['DEM'].squeeze());
![# Welcome to my adventure](BR5MapzenDigitalElevationModel.png)

#### Load in saved eopatch
    load = LoadFromDisk('io_example')
    new_eopatch = load.execute(eopatch_folder='eopatch')

--------------------------
Thoughts: if you compare my first [article](https://danielmoraite.github.io/docs/satellite1.html) on extracting data from a satellite, and then calculate the NDVI and so on.. you'll find that eo-learn can actually help you save tons of time, skipping through a few nerv racking procedures. Plus you got all the eopatches saves for further processing with machine learning. Well done! 

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
  
_________________________
##### Notes:

    # SentinelHub API Key (instance id) stored as an env variable(if you want to call it from your notebook):        
        import os
        os.environ['INSTANCE_ID']='your instance ID'
        INSTANCE_ID = os.environ.get('INSTANCE_ID')



HAVE FUN!

-------------------------
_________________________

