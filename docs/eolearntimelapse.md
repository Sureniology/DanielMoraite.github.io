![# Welcome to my adventure](/images/pandas.gif)

# Satellite Imagery Timelapses


## PANDAS

Summer is here and I keep hearing my dear friend Ron reminding me, regarding my special time and focus on data science, 
how "all work and no play makes jack a dull boy". Well, I haven't got the time yet to properly enjoy the summer, 
though in order to humor him, you and any other data afficionado, I'll have a fun post today! 

And it will be, of course, about pandas, solar farms and timelapses!
 
How I came about it, just by keeping an eye on what happens in the renewables area, and I quote from [article](https://www.businessinsider.com/china-panda-shaped-solar-energy-farms-project-2018-6): 
> In 2017, the groups built a 248-acre solar power plant in Daton, China, that looks from above like two smiling pandas. 
Now the UN, Panda Green Energy, and the Chinese government are on a mission to build 99 more similar solar farms across China.

Looking forward to the future solar farms, meanwhile let's start the fun:

___________________

### Setup 
   - install Sentinel Hub 
   - install eo-learn
   
(please find bellow, under resources, the links for the above)

#### Data Extraction

You'll find details of how to get your area of interest `AOI` coordinates in my previous: [Satellite Imagery Analysis with Python I](https://danielmoraite.github.io/docs/satellite1.html) post. 
Just make sure you select from the menu: `meta`, respectively `add bboxes`. 

Define ROI BBOX and time interval
Request different types of layers and data sources to an eopatch

     roi_bbox = BBox(bbox=[113.4705, 39.9697, 113.4951, 39.9871], crs=CRS.WGS84)
     time_interval = ('2017-02-01', '2017-08-01')

Tasks of the workflow:
     - download S2 images (all 13 bands)
     - filter out images with cloud coverage larger than a given threshold (e.g. 0.8, 0.05)


    layer='BANDS-S2-L1C'
      ​
    wcs_task = S2L1CWCSInput(layer=layer, 
                               resx='5m',
                               resy='5m',
                               maxcc=.05, time_difference=datetime.timedelta(hours=1))
      ​
    save = SaveToDisk('timelapse_example', overwrite_permission=2, compress_level=1)
      
##### Build and execute timelapse as chain of transforms

Feel free to set the max cloud coverage from 0.8 to 0.05 in case the area and timeframes you chose does not have enough 0.8 cc. 
Otherwise it will look as bellow(which is kind of trippy):
![# Welcome to my adventure](/images/withclouds.gif)

    timelapse =LinearWorkflow(wcs_task, save)
      ​
    result = timelapse.execute({
       wcs_task: {'bbox': roi_bbox, 'time_interval': time_interval},
                              save: {'eopatch_folder': 'eopatch'}})
##### Get result as an eopatch

    eopatch = result[save]
    eopatch
 --------------

      EOPatch(
        data: {
          BANDS-S2-L1C: numpy.ndarray(shape=(4, 387, 419, 13), dtype=float32)
        }
        mask: {
          IS_DATA: numpy.ndarray(shape=(4, 387, 419, 1), dtype=bool)
        }
        scalar: {}
        label: {}
        vector: {}
        data_timeless: {}
        mask_timeless: {}
        scalar_timeless: {}
        label_timeless: {}
        vector_timeless: {}
        meta_info: {
          maxcc: 0.05
          service_type: 'wcs'
          size_x: '5m'
          size_y: '5m'
          time_difference: datetime.timedelta(0, 3600)
          time_interval: ('2017-02-01', '2017-08-01')
        }
        bbox: BBox(((113.4705, 39.9697), (113.4951, 39.9871)), crs=EPSG:4326)
        timestamp: [datetime.datetime(2017, 4, 1, 3, 18, 46), ..., datetime.datetime(2017, 7, 30, 3, 21, 38)], length=4
      )
      
##### Function to create GIFs

    import imageio, os
      ​
    def make_gif(eopatch, project_dir, filename, fps):
    """
    Generates a GIF animation from an EOPatch.
    """
    with imageio.get_writer(os.path.join(project_dir, filename), mode='I', fps=fps) as writer:
               for image in eopatch:
                   writer.append_data(np.array(image[..., [3, 2, 1]], dtype=np.uint8))
                      
##### Write EOPatch to GIF

    make_gif(eopatch=eopatch.data['BANDS-S2-L1C']*2.5*255, project_dir='.', filename='eopatch_timelapse1.gif', fps=1)
 

    from IPython.display import Image
    ​
    Image(filename="eopatch_timelapse.gif")


![# Welcome to my adventure](/images/london.gif)


_________________________

Please find the entire code [here](https://github.com/DanielMoraite/DanielMoraite.github.io/blob/master/assets/panda.ipynb)

Feel free to pick you own coordinates. You might want to play with the time frame as well..

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



Feel free to go PAAAANDAAAS!

-------------------------
_________________________
