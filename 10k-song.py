# for C in $(echo -n ABCDEFGHIJKLMNOPQRSTUVWXYZ | sed -e 's/\(.\)/\1\n/g'); do mkdir -p MillionSongSubset-flat/${C}; done
# mmv -c "MillionSongSubset/*/*/*/*.h5" "MillionSongSubset-flat/#1/#4"

# for C in $(echo -n ABCDEFGHIJKLMNOPQRSTUVWXYZ | sed -e 's/\(.\)/\1\n/g'); do mkdir -p MillionSong-flat/${C}; done
# mmv -l "MSD_Data/MSD/snap/data/*/*/*/*.h5" "MillionSong-flat/#1/#4"
# hdfs dfs -D dfs.replication=1 -copyFromLocal MillionSong-flat /courses/datasets/MillionSong-flat
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('million song extractor').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
sc.addPyFile('hdf5_getters_h5py.py')

from hdf5_getters_h5py import *
from msd_schema import msd_schema
# Full dataset
# inputs = '/courses/datasets/MillionSong-flat/*/*'
# h5s = sc.binaryFiles(inputs, minPartitions=100000)

# 10k subset
inputs = '/Users/peterfan/Desktop/MillionSongSubset-flat/*'
# inputs = '/courses/datasets/MillionSongSubset-flat/*/*'
h5s = sc.binaryFiles(inputs, minPartitions=1000)


def extract_hdf5(f):
    h5, filename = open_h5_file(f)

    # TODO: Check against DataFrame generated using original code on the subset.
    
    return Row(
        filename = filename, 
        artist_familiarity = get_artist_familiarity(h5),
        artist_hotttnesss = get_artist_hotttnesss(h5),
        artist_id = get_artist_id(h5),
        artist_mbid = get_artist_mbid(h5),

        artist_playmeid = get_artist_playmeid(h5),
        artist_7digitalid = get_artist_7digitalid(h5),
        artist_latitude = get_artist_latitude(h5),
        artist_longitude = get_artist_longitude(h5),
        artist_location = get_artist_location(h5),

        artist_name = get_artist_name(h5),
        release = get_release(h5),
        release_7digitalid = get_release_7digitalid(h5),
        song_id = get_song_id(h5),
        song_hotttnesss = get_song_hotttnesss(h5),
        
        title = get_title(h5),
        track_7digitalid = get_track_7digitalid(h5),
        similar_artists = get_similar_artists(h5),
        artist_terms = get_artist_terms(h5),
        artist_terms_freq = get_artist_terms_freq(h5),

        artist_terms_weight = get_artist_terms_weight(h5),
        analysis_sample_rate = get_analysis_sample_rate(h5),
        audio_md5 = get_audio_md5(h5),
        danceability = get_danceability(h5),
        duration = get_duration(h5),

        end_of_fade_in = get_end_of_fade_in(h5),
        energy = get_energy(h5),
        key = get_key(h5),
        key_confidence = get_key_confidence(h5),
        loudness = get_loudness(h5),

        mode = get_mode(h5),
        mode_confidence = get_mode_confidence(h5),
        start_of_fade_out = get_start_of_fade_out(h5),
        tempo =  get_tempo(h5),
        time_signature = get_time_signature(h5),

        time_signature_confidence = get_time_signature_confidence(h5),
        track_id = get_track_id(h5),
        segments_start = get_segments_start(h5),
        segments_confidence = get_segments_confidence(h5),
        segments_pitches = get_segments_pitches(h5),

        segments_timbre = get_segments_timbre(h5),
        segments_loudness_max = get_segments_loudness_max(h5),
        segments_loudness_max_time = get_segments_loudness_max_time(h5),
        segments_loudness_start = get_segments_loudness_start(h5),
        sections_start = get_sections_start(h5),

        sections_confidence = get_sections_confidence(h5),
        beats_start = get_beats_start(h5),
        beats_confidence = get_beats_confidence(h5),
        bars_start = get_bars_start(h5),
        bars_confidence = get_bars_confidence(h5),

        tatums_start = get_tatums_start(h5),
        tatums_confidence = get_tatums_confidence(h5),
        artist_mbtags = get_artist_mbtags(h5),
        artist_mbtags_count = get_artist_mbtags_count(h5),
        year = get_year(h5)
    )


data_rdd = h5s.map(extract_hdf5)
data = spark.createDataFrame(data_rdd, schema=msd_schema)
data.write.json('/Users/peterfan/Desktop/msd-10k', mode='overwrite', compression='gzip')
