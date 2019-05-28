![# Welcome to my adventure](/images/sat3.jpeg)

# Satellite Imagery Analysis with Python II


## eo-learn

I guess I have a thing for keeping up with the newest libraries and repositiories for manipumating data from more and more data intensive sources. 
Of course, I wouldn`t have had introduce you to them unless they aide and facilitate the application of machine learning to solve problems and find innovative solutions:
  - efficiently handle big data      - handle different data sources and formats     - complete workflows
  - handle spatio-temporal data      - prototype, build and automate                 - deal with lack of training labels
  - create and validate machine learning applications                                - deal with missing data

Today, I would like to introduce you to an upcoming amazing library: eo-learn. 

eo-learn not only makes extraction of valuable information from satellite imagery easy, is has as well a very high goal: to democratise Earth observation big data.
It started as a response to the availability of open Earth observation (EO) data through the Copernicus and Landsat programs. 
And to quote:
- is a collection of open source Python packages that have been developed to seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner.
- eo-learn library acts as a bridge between Earth observation/Remote sensing field and Python ecosystem for data science and machine learning.

-------------------
In this post will have a look at the basics of using eo-learn and give it a go to downloading, saving and visualizing EO imaging data(RGB, NDVI, Scene Classification masks, digital elevation).


### Setup 
    - install Sentinel Hub 
    - install eo-learn
    - would recommend copying the github repository on your drive
(please find bellow, under resources, the links for the above)

#### Data Extraction

You`ll find details of how to get your area of interest AOI coordinates in my previous: [Satellite Imagery Analysis with Python I](https://danielmoraite.github.io/docs/satellite1.html) post.
Make sure you save the coordinates in a file.geojson in the ../eo-learn-master/example_data/.

        # I have picked an area around my home town, agricultural area. 
          You might play with the time frame as well. 

        
       
        
       
        


And check you download folder. 
_________________________

Please find the entire code [here](https://github.com/DanielMoraite/DanielMoraite.github.io/blob/master/assets/Downloading%20from%20Planet-Copy1.ipynb)
All you'll have to do is pick you own coordinates, instead of spending hour figuring the above. 

> 
_________________________
#### Resources:

  - [documentation for eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html)
  - [eo-learn github](https://github.com/sentinel-hub/eo-learn)
  - [sentinelhub specifications](https://www.sentinel-hub.com/faq/where-get-instance-id)
  - 
        

_________________________
##### Data Trouble Shooting (hopefully you won't need it):

- [Download quota for Open California dataset?](https://gis.stackexchange.com/questions/238803/download-quota-for-open-california-dataset)
- [Why am I unable to activate certain Planet Labs images through the Python API?](https://gis.stackexchange.com/questions/217716/why-am-i-unable-to-activate-certain-planet-labs-images-through-the-python-api/217787) in case you are picking areas from the allowed region of California and still get the innactive feedback. 
        
        # API Key (instance id) stored as an env variable(if you want to call it from your notebook):        
        import os
        os.environ['INSTANCE_ID']='your instance ID'
        INSTANCE_ID = os.environ.get('INSTANCE_ID')



HAVE FUN!

-------------------------


# Exploring the Satellite Imagery: 

## PART II

Time to use python’s Rasterio library since satellite images are grids of pixel-values and can be interpreted as multidimensional arrays.

![# Welcome to my adventure](/images/Cali31.png)

This can be a very useful practice for data preparation for machine & deep learning approached doan the road, including the NDVI which can add to object classification for vegetation (trees, parks, etc.).

Hope you are enjoying it! 

----------------------
----------------------



