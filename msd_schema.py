from pyspark.sql import types

msd_schema = types.StructType([
    types.StructField('filename', types.StringType()),
    types.StructField('artist_familiarity', types.FloatType()),
    types.StructField('artist_hotttnesss', types.FloatType()),
    types.StructField('artist_id', types.StringType()),
    types.StructField('artist_mbid', types.StringType()),
    #5
    types.StructField('artist_playmeid', types.IntegerType()),
    types.StructField('artist_7digitalid', types.IntegerType()),
    types.StructField('artist_latitude', types.FloatType()),
    types.StructField('artist_longitude', types.FloatType()),
    types.StructField('artist_location', types.StringType()),
    #10
    types.StructField('artist_name', types.StringType()),
    types.StructField('release', types.StringType()),
    types.StructField('release_7digitalid', types.IntegerType()),
    types.StructField('song_id', types.StringType()),
    types.StructField('song_hotttnesss', types.FloatType()),
    #15
    types.StructField('title', types.StringType()),
    types.StructField('track_7digitalid', types.IntegerType()),
    types.StructField('similar_artists', types.ArrayType(types.StringType())),
    types.StructField('artist_terms', types.ArrayType(types.StringType())),
    types.StructField('artist_terms_freq', types.ArrayType(types.FloatType())),
    #20
    types.StructField('artist_terms_weight', types.ArrayType(types.FloatType())),
    types.StructField('analysis_sample_rate', types.FloatType()),
    types.StructField('audio_md5', types.StringType()),
    types.StructField('danceability', types.FloatType()),
    types.StructField('duration', types.FloatType()),
    #25
    types.StructField('end_of_fade_in', types.FloatType()),
    types.StructField('energy', types.FloatType()),
    types.StructField('key', types.IntegerType()),
    types.StructField('key_confidence', types.FloatType()),
    types.StructField('loudness', types.FloatType()),
    #30
    types.StructField('mode', types.IntegerType()),
    types.StructField('mode_confidence', types.FloatType()),
    types.StructField('start_of_fade_out', types.FloatType()),
    types.StructField('tempo', types.FloatType()),
    types.StructField('time_signature', types.IntegerType()),    
    #35
    types.StructField('time_signature_confidence', types.FloatType()),
    types.StructField('track_id', types.StringType()),
    types.StructField('segments_start', types.ArrayType(types.FloatType())),
    types.StructField('segments_confidence', types.ArrayType(types.FloatType())),
    types.StructField('segments_pitches', types.ArrayType(types.ArrayType(types.FloatType()))), 
    #40
    types.StructField('segments_timbre', types.ArrayType(types.ArrayType(types.FloatType()))),
    types.StructField('segments_loudness_max', types.ArrayType(types.FloatType())),
    types.StructField('segments_loudness_max_time', types.ArrayType(types.FloatType())),
    types.StructField('segments_loudness_start', types.ArrayType(types.FloatType())),
    types.StructField('sections_start', types.ArrayType(types.FloatType())),
    #45
    types.StructField('sections_confidence', types.ArrayType(types.FloatType())),
    types.StructField('beats_start', types.ArrayType(types.FloatType())),
    types.StructField('beats_confidence', types.ArrayType(types.FloatType())),
    types.StructField('bars_start', types.ArrayType(types.FloatType())),
    types.StructField('bars_confidence', types.ArrayType(types.FloatType())),
    #50
    types.StructField('tatums_start', types.ArrayType(types.FloatType())),
    types.StructField('tatums_confidence', types.ArrayType(types.FloatType())),
    types.StructField('artist_mbtags', types.ArrayType(types.StringType())),
    types.StructField('artist_mbtags_count', types.ArrayType(types.IntegerType())),
    types.StructField('year', types.IntegerType()),
    #55
])