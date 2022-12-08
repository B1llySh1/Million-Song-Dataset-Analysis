# MillionSongClusterAnalysis

A cluster study of Million Song Dataset (http://millionsongdataset.com)

# TODOs
[ ] Update testing instructions when analysis part is finished
[ ] 

## Dependencies

* Python 3.8+
* NumPy
* PySpark 3.2+
* H5Py: https://pypi.org/project/h5py/ 
    - `pip install h5py`
    - `conda install -c anaconda h5py` 
* Seaborn
* Matplotlib


## SFU HDFS Dataset Fillpaths

The flattened HDF5 dataset is on the SFU cluster: `/courses/datasets/MillionSong-flat`
The flattened HDF5 10k subset is in on the SFU cluster: `/courses/datasets/MillionSongSubset-flat`
The extracted JSON.gzip full dataset is on the SFU cluster: `/user/cfa60/msd-full`
The extracted JSON.gzip 10k subset is on the SFU cluster: `/user/cfa60/msd-10k`
The ETL'd CSV intermediate full set is on the SFU cluster: `/user/cfa60/msd-intermediate-full`

## Code Test Run Instructions

Testing must be done on the SFU cluster.

### 1. Testing HDF5 Extractors:
These files converts HDF5 into JSON.gzip using PySpark. H5Py package is required.
* `msd_schema.py` is the Spark dataframe schema used
* `hdf5_getters_h5py.py` contains the getter functions to parse the HDF5 files.
* `10k_song_hdf5_extractor.py` converts the 10k HDF5 subset from MSD
    - Run on SFU Cluster `spark-submit 10k_song_hdf5_extractor.py your_output_dir`.
    - Path to the HDF5 subset on SFU cluster is already in the file.
* `million_song_hdf5_extractor.py` converts the full HDF5 dataset from MSD
    - Path to the HDF5 full dataset on SFU cluster is already in the file.
    - Run on SFU Cluster `spark-submit 10k_song_hdf5_extractor.py your_output_dir`. Note: This takes hours to finish.


### 2. Testing ETL Python Script
This file takes the JSON.gzip generated on the cluster, and perform ETL to generate a CSV intermediate for ML later.
* `msd_kMeans_ETL.py`
    - Run `spark-submit msd_kMeans_ETL.py your_json_gzip_dir your_csv_output_dir`
    - Output will contain 2 subfolders. One for training. One for prediction.

### 3. Testing K-Means Python Script


### 4. Testing PCA Python Script


### 5. Testing Analysis Generation Script