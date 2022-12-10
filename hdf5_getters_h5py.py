""" 
Code adapted to work with Spark on SFU Cluster.
Instead of returning information from a single entry, 
the functions return an array for Spark to form Row().

String objects need to be decoded as UTF-8.

Difference from original code. This does NOT handle aggregate files.
http://millionsongdataset.com/faq/#what-are-song-aggregate-summary-files

"""

import io
import numpy as np
import h5py

# Helper Functions

def open_h5_file(h5_file_root_dir):
    """

    """
    filename, content = h5_file_root_dir
    filename = filename.split('/')[-1]
    h5 = h5py.File(io.BytesIO(content))
    return h5, filename

def decode_str_list(lst):
    """
    convert Iterable[ByteString] to List[str]
    """
    return list(map(lambda b: b.decode('utf-8'), lst))

def decode_float_list(lst):
    """
    convert Iterable[np.float] to Iterable[float]
    """
    return list(map(lambda f: float(f), lst))

def decode_int_list(lst):
    """
    convert Iterable[np.int] to Interable[float]
    """
    return list(map(lambda i: int(i), lst))

def decode_2D_float(lst):
    """
    convert 2D Iterable[np.float] to 2D Interable[float]
    """
    return list(map(lambda x: x.astype(float).tolist(), lst))


# Getter Functions

def get_artist_familiarity(h5):
    """
    Get artist familiarity from a HDF5 song file. Type: float
    """
    return h5['metadata']['songs'][0]['artist_familiarity'].astype(float).tolist()

def get_artist_hotttnesss(h5):
    """
    Get artist hotttnesss from a HDF5 song file. Type: float.
    """
    return h5['metadata']['songs'][0]['artist_hotttnesss'].astype(float).tolist()

def get_artist_id(h5):
    """
    Get artist id from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['artist_id'].decode('utf-8')

def get_artist_mbid(h5):
    """
    Get artist musibrainz id from a HDF5 song file. Type: string.
    """
    return h5['metadata']['songs'][0]['artist_mbid'].decode('utf-8')

def get_artist_playmeid(h5):
    """
    Get artist playme id from a HDF5 song file. Type: int
    """
    return h5['metadata']['songs'][0]['artist_playmeid'].tolist()

def get_artist_7digitalid(h5):
    """
    Get artist 7digital id from a HDF5 song file. Type: int
    """
    return h5['metadata']['songs'][0]['artist_7digitalid'].tolist()

def get_artist_latitude(h5):
    """
    Get artist latitude from a HDF5 song file. Type: float
    """
    return h5['metadata']['songs'][0]['artist_latitude'].astype(float).tolist()

def get_artist_longitude(h5):
    """
    Get artist longitude from a HDF5 song file. Type: float
    """
    return h5['metadata']['songs'][0]['artist_longitude'].astype(float).tolist()

def get_artist_location(h5):
    """
    Get artist location from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['artist_location'].decode('utf-8')

def get_artist_name(h5):
    """
    Get artist name from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['artist_name'].decode('utf-8')

def get_release(h5):
    """
    Get release from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['release'].decode('utf-8')

def get_release_7digitalid(h5):
    """
    Get release 7digital id from a HDF5 song file. Type: int
    """
    return h5['metadata']['songs'][0]['release_7digitalid'].tolist()

def get_song_id(h5):
    """
    Get song id from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['song_id'].decode('utf-8')

def get_song_hotttnesss(h5):
    """
    Get song hotttnesss from a HDF5 song file. Type: float
    """
    return h5['metadata']['songs'][0]['song_hotttnesss'].astype(float).tolist()

def get_title(h5):
    """
    Get title from a HDF5 song file. Type: string
    """
    return h5['metadata']['songs'][0]['title'].decode('utf-8')

def get_track_7digitalid(h5):
    """
    Get track 7digital id from a HDF5 song file. Type: int
    """
    return h5['metadata']['songs'][0]['track_7digitalid'].tolist()

def get_similar_artists(h5):
    """
    Get similar artists array. Type: array string
    """
    return decode_str_list(h5['metadata']['similar_artists'])

def get_artist_terms(h5):
    """
    Get artist terms array. Type: array string
    """
    return decode_str_list(h5['metadata']['artist_terms'])


def get_artist_terms_freq(h5):
    """
    Get artist terms array frequencies. Type: array float
    """
    return decode_float_list(h5['metadata']['artist_terms_freq'])

def get_artist_terms_weight(h5):
    """
    Get artist terms array frequencies. Type: array float
    """
    return decode_float_list(h5['metadata']['artist_terms_weight'])

def get_analysis_sample_rate(h5):
    """
    Get analysis sample rate from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['analysis_sample_rate'].astype(float).tolist()

def get_audio_md5(h5):
    """
    Get audio MD5 from a HDF5 song file. Type: string
    """
    return h5['analysis']['songs'][0]['audio_md5'].decode('utf-8')

def get_danceability(h5):
    """
    Get danceability from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['danceability'].astype(float).tolist()

def get_duration(h5):
    """
    Get duration from a HDF5 song file/ Type: float
    """
    return h5['analysis']['songs'][0]['duration'].astype(float).tolist()

def get_end_of_fade_in(h5):
    """
    Get end of fade in from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['end_of_fade_in'].astype(float).tolist()

def get_energy(h5):
    """
    Get energy from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['energy'].astype(float).tolist()

def get_key(h5):
    """
    Get key from a HDF5 song file. Type: int
    """
    return h5['analysis']['songs'][0]['key'].tolist()

def get_key_confidence(h5):
    """
    Get key confidence from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['key_confidence'].astype(float).tolist()

def get_loudness(h5):
    """
    Get loudness from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['loudness'].astype(float).tolist()

def get_mode(h5):
    """
    Get mode from a HDF5 song file. Type: int
    """
    return h5['analysis']['songs'][0]['mode'].tolist()

def get_mode_confidence(h5):
    """
    Get mode confidence from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['mode_confidence'].astype(float).tolist()

def get_start_of_fade_out(h5):
    """
    Get start of fade out from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['start_of_fade_out'].astype(float).tolist()

def get_tempo(h5):
    """
    Get tempo from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['tempo'].astype(float).tolist()

def get_time_signature(h5):
    """
    Get signature from a HDF5 song file. Type: int
    """
    return h5['analysis']['songs'][0]['time_signature'].tolist()

def get_time_signature_confidence(h5):
    """
    Get signature confidence from a HDF5 song file. Type: float
    """
    return h5['analysis']['songs'][0]['time_signature_confidence'].astype(float).tolist()

def get_track_id(h5):
    """
    Get track id from a HDF5 song file. Type: string
    """
    return h5['analysis']['songs'][0]['track_id'].decode('utf-8')

def get_segments_start(h5):
    """
    Get segments start array. Type: array float
    """
    return decode_float_list(h5['analysis']['segments_start'])
    
def get_segments_confidence(h5):
    """
    Get segments confidence array. Type: array float
    """
    return decode_float_list(h5['analysis']['segments_confidence'])


def get_segments_pitches(h5):
    """
    Get segments pitches array. Type: 2D array float
    """
    return decode_2D_float(h5['analysis']['segments_pitches'])

def get_segments_timbre(h5):
    """
    Get segments timbre array. Type: 2D array float
    """
    return decode_2D_float(h5['analysis']['segments_timbre'])

def get_segments_loudness_max(h5):
    """
    Get segments loudness max array. Type: array float
    """
    return decode_float_list(h5['analysis']['segments_loudness_max'])


def get_segments_loudness_max_time(h5):
    """
    Get segments loudness max time array. Type: array float
    """
    return decode_float_list(h5['analysis']['segments_loudness_max_time'])

def get_segments_loudness_start(h5):
    """
    Get segments loudness start array. Type: array float
    """
    return decode_float_list(h5['analysis']['segments_loudness_start'])

def get_sections_start(h5):
    """
    Get sections start array. Type: array float
    """
    return decode_float_list(h5['analysis']['sections_start'])

def get_sections_confidence(h5):
    """
    Get sections confidence array. Type: array float
    """
    return decode_float_list(h5['analysis']['sections_confidence'])

def get_beats_start(h5):
    """
    Get beats start array. Type: array float
    """
    return decode_float_list(h5['analysis']['beats_start'])

def get_beats_confidence(h5):
    """
    Get beats confidence array. Type: array float
    """
    return decode_float_list(h5['analysis']['beats_confidence'])


def get_bars_start(h5):
    """
    Get bars start array. Type: array float
    """
    return decode_float_list(h5['analysis']['bars_start'])

def get_bars_confidence(h5):
    """
    Get bars start array. Type: array float
    """
    return decode_float_list(h5['analysis']['bars_confidence'])

def get_tatums_start(h5):
    """
    Get tatums start array. Type: array float
    """
    return decode_float_list(h5['analysis']['tatums_start'])

def get_tatums_confidence(h5):
    """
    Get tatums confidence array. Type: array float
    """
    return decode_float_list(h5['analysis']['tatums_confidence'])

def get_artist_mbtags(h5):
    """
    Get artist musicbrainz tag array. Type: array string
    """
    return decode_str_list(h5['musicbrainz']['artist_mbtags'])

def get_artist_mbtags_count(h5):
    """
    Get artist musicbrainz tag count array. Type: array int
    """
    return decode_int_list(h5['musicbrainz']['artist_mbtags_count'])

def get_year(h5):
    """
    Get release year from a HDF5 song file. Type: int
    """
    return h5['musicbrainz']['songs'][0]['year'].tolist()
