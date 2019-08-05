__author__      = "Aleksandr Diment"
__email__       = "aleksandr.diment@tut.fi"

import yaml, os, sys, shutil, librosa, soundfile as sf, numpy as np, pandas as pd
import argparse, textwrap
from IPython import embed
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from tqdm import tqdm
import hashlib
import warnings

classnames = ['knock', 'keys', 'clearthroat']

common_fs = 44100
bitdepth = 24  # originals can be whatever, but after mixing and lowering the levels, we'd like more precision
BG_LENGTH_SECONDS = 5
magic_anticlipping_factor = 0.2
subsets = ['devtrain', 'devtest']
default_params = 'mixing_params_devtest_dcase_fixed.yaml'
default_param_hash = '20b255387a2d0cddc0a3dff5014875e7'


def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data


def write_meta_yaml(filename, data):
    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(data,default_flow_style=False))


def list_audio_files(folder):
    files = []
    for dirpath,d,f in os.walk(folder):
        for file in f:
            if file[-4:].lower()=='.wav' or file[-5:].lower()=='.flac':
                files.append(os.path.join(dirpath,file))
    return files


def load_audio(path, target_fs=None):
    """
    Reads audio with (currently only one supported) backend (as opposed to librosa, supports more formats and 32 bit
    wavs) and resamples it if needed
    :param path: path to wav/flac/ogg etc (http://www.mega-nerd.com/libsndfile/#Features)
    :param target_fs: if None, original fs is kept, otherwise resampled
    :return:
    """
    y, fs = sf.read(path)
    if y.ndim>1:
        y = np.mean(y, axis=1)
    if target_fs is not None and fs!=target_fs:
        #print('Resampling %d->%d...' %(fs, target_fs))
        y = librosa.resample(y, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return y, fs


def write_audio(path, audio, fs, bitdepth=None):
    ext = os.path.splitext(path)[1].lower()
    if bitdepth==None:
        sf.write(file=path, data=audio, samplerate=fs)  # whatever is default in soundfile
    if bitdepth==24 and (ext=='.wav' or ext=='.flac'):
        #print('Writing 24 bits pcm, yay!')
        sf.write(file=path, data=audio, samplerate=fs, subtype='PCM_24')
    elif bitdepth==32:
        if ext=='.wav':
            sf.write(file=path, data=audio, samplerate=fs, subtype='PCM_32')
        else:
            print('Writing into {} format with bit depth {} is not supported, reverting to the default {}'.format(
                ext, bitdepth, sf.default_subtype(ext[1:])))
            sf.write(file=path, data=audio, samplerate=fs)
    elif bitdepth==16:
            sf.write(file=path, data=audio, samplerate=fs, subtype='PCM16')
    else:
        raise IOError('Unexpected bit depth {}'.format(bitdepth))


def freesound_files_into_event_wise_files(input_path, output_path):
    """
    Not used in the final implementation, but might be useful for some. Isolates the annotated events from the original
    freesound files using the accompanied yaml meta-files into per-event audio files.
    :param input_path: path to directory with folders=classes, inside - wav files of any names with corresponding yaml
    files of the same names.
    :param output_path: where to put the chunked audios in structure classnames->events
    :return:
    """
    for classname in classnames:
        # get all files
        current_folder = os.path.join(input_path, classname)
        audio_files = list_audio_files(current_folder)
        meta_files = [filepath.replace('wav','yaml') for filepath in audio_files]
        for meta_file, audio_file in zip(meta_files, audio_files):
            # read meta
            meta = read_meta_yaml(meta_file)
            # read audio
            y,fs = load_audio(audio_file, target_fs=common_fs)
            for id, segment in enumerate(meta['valid_segments']):
                segment_audio = y[int(round(segment[0]*fs)):int(round(segment[1]*fs))]
                output_filepath = audio_file.replace(input_path, output_path).replace('.wav', '_%d.wav' %id)
                output_filefolder = os.path.split(output_filepath)[0]
                if not os.path.exists(output_filefolder):
                    os.makedirs(output_filefolder)
                print('Writing audio into %s' %output_filepath)
                write_audio(output_filepath, segment_audio, fs, bitdepth=bitdepth)


def rmse(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_event_amplitude_scaling_factor(s, n, target_ebr_db, method='rmse'):
    """
    Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
    and rmse is calculated over the whole signal
    """
    original_sn_rmse_ratio = rmse(s) / rmse(n)
    target_sn_rmse_ratio =  10 ** (target_ebr_db / float(20))
    signal_scaling_factor = target_sn_rmse_ratio/original_sn_rmse_ratio
    return signal_scaling_factor


def mix(bg_audio, event_audio, event_offset_samples, scaling_factor, magic_anticlipping_factor):
    """
    Mix np arrays of background and event audio (mono, non-matching lengths supported, sampling frequency better be the
    same, no operation in terms of seconds is performed though)
    :param bg_audio:
    :param event_audio:
    :param event_offset_samples:
    :param scaling_factor:
    :return:
    """

    old_event_audio = event_audio
    event_audio = scaling_factor*event_audio
    # check that the offset is not too long
    longest_possible_offset = len(bg_audio) - len(event_audio)
    print(longest_possible_offset, event_offset_samples)
    if event_offset_samples > longest_possible_offset:
        raise AssertionError('Wrongly generated event offset: event tries to go outside the boundaries of the bg')
        #event_offset_samples = longest_possible_offset # shouldn't really happen if we pregenerate offset accounting for the audio lengths

    # measure how much to pad from the right
    tail_length = len(bg_audio) - len(event_audio) - event_offset_samples
    # pad zeros at the beginning of event signal
    padded_event = np.pad(event_audio, pad_width=((event_offset_samples, tail_length)), mode='constant', constant_values=0)
    if not len(padded_event)==len(bg_audio):
        raise AssertionError('Mixing yielded a signal of different length than bg! Should not happen')
    mixture = magic_anticlipping_factor* (padded_event + bg_audio)
    # Done! Now let's just confirm lengths mach
    # Also nice to make sure that we did not introduce clipping
    if np.max(np.abs(mixture)) >= 1:
        normalisation_factor = 1/float(np.max(np.abs(mixture)))
        print('Attention! Had to normalise the mixture by * %f' %normalisation_factor)
        print('I.e. bg max: %f, event max: %f, sum max: %f'
              %(np.max(np.abs(bg_audio)), np.max(np.abs(padded_event)), np.max(np.abs(mixture))))
        mixture /= np.max(np.abs(mixture))
        print('The scaling factor for the event was %f' %scaling_factor)
        print('The event before scaling was max %f' %np.max(np.abs(old_event_audio)))
    # now also refine the start time for the annotation
    return mixture #, start_time_seconds, end_time_seconds


def get_dict_of_files(path):
    # Assume that this is called separately for train/test. That the split is on a higher level
    # path contains: folders = event/bg class names, inside each collection of wav/flac files
    allfiles = np.array([])
    classes = next(os.walk(path))[1]
    for class_id, classname in enumerate(classes):
        cur_class_path = os.path.join(path, classname)
        # ls wavs/flacs
        cur_class_files = list_audio_files(cur_class_path)
        # collect to list of events: probably just paths
        for cur_file in cur_class_files:
            cur_entry = {'filepath': cur_file, 'classid': class_id, 'classname': classname}
            allfiles = np.append(allfiles,cur_entry)
    return allfiles


def generate_mixture_recipes(data_path=os.path.join('..','data'), current_subsets=np.array(['devtrain', 'devtest']), mixing_params=None):

    if mixing_params is not None:
        ebrs = mixing_params['ebrs']
        MIXTURES = mixing_params['mixtures']
        max_events_in_mixture = mixing_params['max_events_in_mixture']
        event_prob = mixing_params['event_prob']
        seed = mixing_params['seed']
    else:
        ebrs = [-6,0,6]
        MIXTURES = 500
        max_events_in_mixture = 10
        event_prob = 0.8
        seed = 42

    m = hashlib.md5
    param_hash = m(yaml.dump(mixing_params).encode('utf-8')).hexdigest()
    print('Param hash: {}'.format(param_hash))

    bgs_path = os.path.join(data_path, 'source_data', 'bgs')
    events_path = os.path.join(data_path, 'source_data', 'events')
    cv_setup_path = os.path.join(data_path, 'source_data', 'cv_setup')
    mixture_data_path = os.path.join(data_path, 'mixture_data')
    for subset in current_subsets:
        if not os.path.exists(os.path.join(mixture_data_path,subset,param_hash)):
            os.makedirs(os.path.join(mixture_data_path,subset,param_hash))

    print('Seed: {}'.format(seed))
    
    for subset in reversed(current_subsets):
        r = np.random.RandomState(seed)
        #print('Current subset: %s' %subset)
        try:
            classwise_events = read_meta_yaml(os.path.join(cv_setup_path,'events_%s.yaml' %subset))
            allbgs = read_meta_yaml(os.path.join(cv_setup_path,'bgs_%s.yaml' %subset))
        except IOError:
            sys.exit('Failed to load source data from {}, check the provided path. Quitting.'.format(cv_setup_path))
        
        mixture_recipes = []
        
        bgs = r.choice(allbgs, MIXTURES)
        
        for mixture_id in range(MIXTURES):
            print('Current Mixture: {}'.format(mixture_id))
            event_presence_flag = 0
            bg = bgs[mixture_id]
            
            p1 = event_prob/np.sum([0.5**i for i in range(max_events_in_mixture)])
            p = [1-event_prob] + [p1/2**i for i in range(max_events_in_mixture)]
            
            number_of_events = r.choice(np.arange(max_events_in_mixture + 1), p = p)
#            print('Number of events ', number_of_events)
            cur_events = []
            for event_id in range(number_of_events):
                event_presence_flag = 1
                
#                print('Current Event: {}'.format(event_id))
                classname = r.choice(classnames)
                cur_event = r.choice(classwise_events[classname])
                cur_event['classname'] = classname
                cur_event['audio_filepath'] = os.path.join(classname, cur_event['audio_filename'])
                cur_event['length_seconds'] = np.diff(cur_event['segment'])[0]
                cur_event['offset_seconds'] = (BG_LENGTH_SECONDS-cur_event['length_seconds'])*r.rand()
                
                cur_events.append(cur_event)
                    
#        for classname in classnames:
#           print('Current class: {}'.format(classname))
#           curclassevents = []
#            cur_events = classwise_events[classname]
#            for event in cur_events:
#                event['classname'] = classname
#                event['audio_filepath'] = os.path.join(classname, event['audio_filename'])
#                event['length_seconds'] = np.diff(event['segment'])[0]
#                curclassevents.append(event)

#            events = r.choice(curclassevents, int(round(MIXTURES*event_presence_prob)) )

#            event_presence_flags = (np.hstack((np.ones(len(events)), np.zeros(len(bgs)-len(events))))).astype(bool)
#            event_presence_flags = r.permutation(event_presence_flags)
#            event_instance_ids = np.nan*np.ones(len(bgs)).astype(int)  # by default event id set to nan: no event. fill it later with actual event ids when needed
#            event_instance_ids[event_presence_flags] = np.arange(len(events))

#            for event in events:
#                event['offset_seconds'] = (BG_LENGTH_SECONDS-event['length_seconds'])*r.rand()

            event_starts_in_mixture_seconds = np.nan * np.ones(len(cur_events))
            event_starts_in_mixture_seconds = [event['offset_seconds'] for event in cur_events]

            # double-check that we didn't shuffle things wrongly: check that the offset never exceeds bg_len-event_len
            checker = [offset+event['length_seconds'] for offset, event in
                       zip(event_starts_in_mixture_seconds, cur_events)]
            if len(checker) != 0:
                assert np.max(np.array(checker)) < BG_LENGTH_SECONDS

            ebr = r.choice(ebrs)

            # for recipes, we gotta provide amplitude scaling factors instead of ebrs: the latter are more ambiguous
            # so, go through files, measure levels, calculate scaling factors
            mixture_recipe = {}
            
            mixture_recipe['bg_path'] = bg['filepath']
            mixture_recipe['bg_classname'] = bg['classname']
            mixture_recipe['event_present'] = bool(event_presence_flag)
            if event_presence_flag:
                mixture_recipe['ebr'] = float(ebr)

                assert len(cur_events) != 0  # shouldn't happen, nans are in sync with falses in presence flags

                try:
                    bg_audio, fs_bg = load_audio(os.path.join(bgs_path,bg['filepath']), target_fs=common_fs)
                except:
                    embed()
                mixture_recipe_events = []
                for event_id, event in enumerate(cur_events):
                    event_in_mixture_recipe = {}
                    event_audio, fs_event = load_audio(os.path.join(events_path,event['audio_filepath']), target_fs=common_fs)
                    assert fs_bg==fs_event, 'Fs mismatch! Expected resampling taken place already'
                    fs = fs_event
                    segment_start_samples = int(event['segment'][0] * fs)
                    segment_end_samples = int(event['segment'][1] * fs)
                    event_audio = event_audio[segment_start_samples:segment_end_samples]

                    # let's calculate the levels of bgs also at the location of the event only
                    eventful_part_of_bg = bg_audio[int(event_starts_in_mixture_seconds[event_id]*fs):int(event_starts_in_mixture_seconds[event_id]*fs+len(event_audio))]
                    scaling_factor = get_event_amplitude_scaling_factor(event_audio, eventful_part_of_bg, target_ebr_db=ebr)

                    event_in_mixture_recipe['event_path'] = event['audio_filepath']
                    event_in_mixture_recipe['event_class'] = str(event['classname'])
                    event_in_mixture_recipe['event_start_in_mixture_seconds'] = float(event_starts_in_mixture_seconds[event_id])
                    event_in_mixture_recipe['event_length_seconds'] = float(event['length_seconds'])
                    event_in_mixture_recipe['scaling_factor'] = float(scaling_factor)
                    event_in_mixture_recipe['segment_start_seconds'] = event['segment'][0]
                    event_in_mixture_recipe['segment_end_seconds'] = event['segment'][1]
                    
                    mixture_recipe_events.append(event_in_mixture_recipe)
                mixture_recipe['events_recipes'] = mixture_recipe_events
#                print(mixture_recipe_events)
                
            else:
                mixture_recipe['ebr'] = -np.inf
                
            
                
            mixing_param_hash = m(yaml.dump(mixture_recipe).encode('utf-8')).hexdigest()
            classnames_str = set([event['classname'] for event in cur_events])
            mixture_recipe['mixture_audio_filename'] = 'mixture' + '_' + subset + '_' + '%03d' % mixture_id + '_' + '_'.join(classnames_str ) + '_' + mixing_param_hash + '.wav'
            if event_presence_flag:
                mixture_recipe['annotation_string'] = mixture_recipe['mixture_audio_filename'] + '\t' + str(len(mixture_recipe_events)) \
                                                       + '\t' + '\t'.join(['\t'.join([str(event_recipe['event_class']),
                                                                           str(event_recipe['event_start_in_mixture_seconds']),
                                                                           str(event_recipe['event_length_seconds'])
                                                                           ]) for event_recipe in mixture_recipe_events])
    #                                                  + '\t' + str(mixture_recipe['event_start_in_mixture_seconds']) \
    #                                                  + '\t' + str(mixture_recipe['event_start_in_mixture_seconds']
    #                                                               +mixture_recipe['event_length_seconds']) \
    #                                                  + '\t' + mixture_recipe['event_class']
            else:
                mixture_recipe['annotation_string'] = mixture_recipe['mixture_audio_filename'] + '\t' + str(0) #+ '\t' + 'None' + '\t0\t30'
            mixture_recipes.append(mixture_recipe)
            cur_recipe_folder = os.path.join(mixture_data_path,subset,param_hash,'meta')
            if not os.path.exists(cur_recipe_folder):
                os.makedirs(cur_recipe_folder)
            write_meta_yaml(os.path.join(cur_recipe_folder, 'mixture_recipes_' + subset + '.yaml'), mixture_recipes)
            #yaml.safe_dump(mixture_recipes)
            print('Mixture recipes dumped into file {} successfully'.format(os.path.join(cur_recipe_folder, 'mixture_recipes_' + subset +'.yaml')))
            print('-'*20)



def bg_dcase_annotations_preprocess(data_path):
    input_path = os.path.join(data_path,'source_data','bgs')
    output_path = os.path.join(data_path,'source_data','cv_setup')
    meta_filepath = os.path.join(input_path,'meta.txt')
    error_filepath = os.path.join(input_path,'error.txt')
    bg_screening_filepath = os.path.join(input_path, 'bg_screening.csv')
    df=pd.read_csv(meta_filepath, sep='\t',header=None)
    err = pd.read_csv(error_filepath, sep='\t', header=None)
    erroneous_files = [os.path.split(e[0])[-1] for e in  err.values]
    screening_df = pd.read_csv(bg_screening_filepath, sep=',')
    screening_skips = [scr[0] for scr in screening_df.values if not np.all(pd.isnull(scr[1:4]))]
    metas = []
    for entry in df.values:
        if (not os.path.split(entry[0])[-1] in erroneous_files) and (not os.path.split(entry[0])[-1] in screening_skips):
            input_filepath = entry[0]
            meta = {}
            meta['filepath'] = input_filepath
            meta['classname'] = entry[1]
            metas.append(meta)
        else:
            print('Skipping erroneous file %s' %entry[0])
            if os.path.exists(os.path.join(input_path,entry[0])):
                print('Also, deleting the physical file')
                os.remove(os.path.join(input_path,entry[0]))
    unique_classnames = np.sort(np.unique([meta['classname'] for meta in metas])).tolist()
    metas_with_classids = [{'filepath':meta['filepath'],
                            'classname':meta['classname'],
                            'classid':unique_classnames.index(meta['classname'])}
                           for meta in metas]

    train_bgs = []
    dev_bgs = []

    # splits can be done in terms of classes, but currently not. using the original dcase 2016 fold 1
    fold1_train = [f[0] for f in pd.read_csv(os.path.join(input_path,'fold1_train.txt'), sep='\t', header=None).values]
    fold1_evaluate = [f[0] for f in pd.read_csv(os.path.join(input_path,'fold1_evaluate.txt'), sep='\t', header=None).values]
    for meta in metas_with_classids:
        if meta['filepath'] in fold1_train:
            train_bgs.append(meta)
        elif meta['filepath'] in fold1_evaluate:
            dev_bgs.append(meta)
        else:
            raise IOError('File %s not found in either train or evaluate files of dcase2016, weird' %meta['filepath'])

    write_meta_yaml(os.path.join(output_path,'bgs_devtest.yaml'), dev_bgs)
    write_meta_yaml(os.path.join(output_path,'bgs_devtrain.yaml'), train_bgs)

def bg_dcase_evaltest_annotations_preprocess(data_path):
    input_path = os.path.join(data_path,'source_data','bgs')
    output_path = os.path.join(data_path,'source_data','cv_setup')
    meta_filepath = os.path.join(input_path,'meta.txt')
    bg_screening_filepath = os.path.join(input_path, 'bg_screening.csv')
    df=pd.read_csv(meta_filepath, sep='\t',header=None)
    screening_df = pd.read_csv(bg_screening_filepath, sep=',')
    screening_skips = [scr[0] for scr in screening_df.values if not np.all(pd.isnull(scr[1:4]))]
    print('Screening skips: ')
    print(screening_skips)
    metas = []
    for entry in df.values:
        if (not os.path.split(entry[0])[-1] in screening_skips):
            input_filepath = entry[0]
            meta = {}
            meta['filepath'] = input_filepath
            meta['classname'] = entry[1]
            metas.append(meta)
        else:
            print('Skipping erroneous file %s' %entry[0])
            if os.path.exists(os.path.join(input_path,entry[0])):
                print('Also, deleting the physical file')
                os.remove(os.path.join(input_path,entry[0]))
    unique_classnames = np.sort(np.unique([meta['classname'] for meta in metas])).tolist()
    metas_with_classids = [{'filepath':meta['filepath'],
                            'classname':meta['classname'],
                            'classid':unique_classnames.index(meta['classname'])}
                           for meta in metas]

    evaltest_bgs = []
    # splits can be done in terms of classes, but currently not. using the original dcase 2016 fold 1
    for meta in metas_with_classids:
        evaltest_bgs.append(meta)
    write_meta_yaml(os.path.join(output_path,'bgs_evaltest.yaml'), evaltest_bgs)


def relocate_bg_data(input_path='bg_data', output_path = 'newdata'):
    meta_filepath = os.path.join(input_path,'meta.txt')
    error_filepath = os.path.join(input_path,'error.txt')
    df=pd.read_csv(meta_filepath, sep='\t',header=None)
    err = pd.read_csv(error_filepath, sep='\t', header=None)
    erroneous_files = [e[0] for e in  err.values]

    for entry in df.values:
        output_directory = os.path.join(output_path, entry[1])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if not entry[0] in erroneous_files:
            input_filepath = os.path.join(input_path, entry[0])
            output_filename = entry[0].replace('audio/','')
            shutil.copy(input_filepath, os.path.join(output_directory, output_filename))
        else:
            print('Skipping erroneous file %s' %entry[0])


def do_mixing(data_path =os.path.join('..','data'), current_subsets = np.array(['devtrain', 'devtest']), magic_anticlipping_factor=0.2, param_hash=default_param_hash):

    bgs_path = os.path.join(data_path,'source_data', 'bgs')
    events_path = os.path.join(data_path,'source_data', 'events')
    mixture_data_path = os.path.join(data_path,'mixture_data')
    for subset in current_subsets:
        if not os.path.exists(os.path.join(mixture_data_path,subset,param_hash)):
            os.makedirs(os.path.join(mixture_data_path,subset,param_hash))

    # Mix according to the recipe
    for subset in current_subsets:
        print('Current subset: {}'.format(subset))
        print(os.path.join(mixture_data_path,subset,param_hash,'meta', 'mixture_recipes_' + subset + '.yaml'))
        mixture_recipes = read_meta_yaml(os.path.join(mixture_data_path,subset,param_hash,'meta', 'mixture_recipes_' + subset + '.yaml'))
        #mixture_recipes = [mixture_recipe for mixture_recipe in mixture_recipes if mixture_recipe['mixture_class']==classname]
        for mixture_recipe in tqdm(mixture_recipes):
            bg_path_full = os.path.join(bgs_path, mixture_recipe['bg_path'])
            bg_audio, fs_bg = load_audio(bg_path_full, target_fs=common_fs)
            if mixture_recipe['event_present']:
                for event_recipe in mixture_recipe['events_recipes']:
                
                    event_path_full = os.path.join(events_path, event_recipe['event_path'])
                    event_audio, fs_event = load_audio(event_path_full, target_fs=common_fs)
                    assert fs_bg==fs_event and fs_bg==common_fs, 'Fs mismatch! Expected resampling taken place already'
                    segment_start_seconds = event_recipe['segment_start_seconds']
                    segment_end_seconds = event_recipe['segment_end_seconds']
                    segment_start_samples = int(segment_start_seconds*common_fs)
                    segment_end_samples = int(segment_end_seconds*common_fs)
                    event_audio = event_audio[segment_start_samples:segment_end_samples]
                    event_start_in_mixture_seconds = event_recipe['event_start_in_mixture_seconds']
                    print(event_start_in_mixture_seconds)
                    event_start_in_mixture_samples = int(event_start_in_mixture_seconds*common_fs)
                    scaling_factor = event_recipe['scaling_factor']
                    mixture = mix(bg_audio, event_audio, event_start_in_mixture_samples, scaling_factor=scaling_factor, magic_anticlipping_factor=magic_anticlipping_factor)
                    bg_audio = mixture
            else:
                mixture = magic_anticlipping_factor*bg_audio
            current_audio_folder_path = os.path.join(mixture_data_path,subset,param_hash,'audio')
            if not os.path.exists(current_audio_folder_path):
                os.makedirs(current_audio_folder_path)
            write_audio(os.path.join(current_audio_folder_path,mixture_recipe['mixture_audio_filename']), mixture, common_fs, bitdepth=bitdepth)
        # also dump annotation strings into a txt file in the submission format
        annotation_strings = [mixture_recipe['annotation_string'] for mixture_recipe in mixture_recipes]
        np.savetxt(os.path.join(mixture_data_path,subset,param_hash,'meta','event_list_' + subset + '.csv'), annotation_strings, fmt='%s')


def find_surroundings_of_a_peak(audio, chunk_length):
    peak = np.argmax(np.abs(audio))
    if peak-0.1*chunk_length >=0 and peak+0.9*chunk_length <= len(audio):
        return audio[int(peak-0.1*chunk_length):int(peak+0.9*chunk_length)]
    elif peak+0.9*chunk_length > len(audio) and len(audio)-chunk_length >=0:
        return audio[-1-int(chunk_length):-1]
    elif chunk_length <= len(audio):
        return audio[0:int(chunk_length)]
    else:
        return audio

def calculate_hash_of_audio_filenames_from_folder(path):
    audio_filepaths = list_audio_files(path)
    audio_filenames = [os.path.split(path)[-1] for path in audio_filepaths]
    m = hashlib.md5
    hash = m(yaml.dump(audio_filenames).encode('utf-8')).hexdigest()
    return hash

def get_mixing_params(filename='mixing_params.yaml'):
    try:
        mixing_params = read_meta_yaml(filename)
    except IOError:
        print('Failed to load specified params from {}, rolling back to default {}'.format(filename,default_params))
        try:
            mixing_params = read_meta_yaml(default_params)
        except IOError:
            sys.exit('Default params not found either, dataset seems corrupted, cannot work. Bye.')
    return mixing_params


def main(data_path=os.path.join('..','data'),
         generate_devtrain_recipes=True, generate_devtest_recipes=False, generate_evaltest_recipes=False,
         devtrain_mixing_param_file='mixing_params.yaml',
         devtest_mixing_param_file='mixing_params_devtest_dcase_fixed.yaml',
         evaltest_mixing_param_file='mixing_params_evaltest.yaml',
         synthesize_devtrain_mixtures=True, synthesize_devtest_mixtures=False, synthesize_evaltest_mixtures=False):

    if generate_evaltest_recipes:
        print('Let us clean up bgs...')
        bg_dcase_evaltest_annotations_preprocess(data_path=data_path)
        print('Done!')
    elif generate_devtrain_recipes or generate_devtest_recipes:
        print('Let us clean up bgs...')
        bg_dcase_annotations_preprocess(data_path=data_path)
        print('Done!')

    data_path_def = os.path.join('..','data')  # there is one default value in the function
    # signature, however, we need to also revert to the default if it's overridden

    # # TODO implement downloading the source data from the internets.
    # if generate_devtest_recipes or generate_devtrain_recipes or synthesize_devtrain_mixtures or synthesize_devtest_mixtures:
    #     if not os.path.exists(data_path):
    #         try:
    #             os.makedirs(data_path)
    #         except OSError:
    #             print('Tried creating the specified path {}, failed, rolling back to a default {}'.format(data_path, data_path_def))
    #             data_path = data_path_def
    #             try:
    #                 os.makedirs(data_path)
    #             except OSError:
    #                 print('Failed to create data folder in the default location {} either. Giving up...'.format(data_path))
    #                 sys.exit()


    m = hashlib.md5

    if generate_devtrain_recipes:
        print('='*30)
        print('Generating devtrain mixture recipes...')
        mixing_params = get_mixing_params(devtrain_mixing_param_file)
        generate_mixture_recipes(data_path=data_path, current_subsets=np.array(['devtrain']), mixing_params=mixing_params)
    if generate_devtest_recipes:
        print('='*30)
        print('Generating devtest mixture recipes...')
        mixing_params = get_mixing_params(devtest_mixing_param_file)

        hash = m(yaml.dump(mixing_params).encode('utf-8')).hexdigest()

        if hash != '20b255387a2d0cddc0a3dff5014875e7':
            if devtest_mixing_param_file == 'mixing_params_devtest_dcase_fixed.yaml':
                warnings.warn('We detected a change in mixing params file {}. It is supposed to be fixed if you wish to '
                              'replicate the DCASE devtest subset! If you wish to have your own devtest subset, '
                              'you should better specify the params in mixing_params_devtest.yaml file and use '
                              'it.'.format(devtest_mixing_param_file))
            else:
                print('We noticed that you are using devtest mixing params different from the DCASE fixed ones. '
                      'Please remember to do evaluation on the devtest_dcase_fixed set when you report the results.')
        generate_mixture_recipes(data_path=data_path, current_subsets=np.array(['devtest']), mixing_params=mixing_params)
    if generate_evaltest_recipes:
        print('='*30)
        print('Generating evaltest mixture recipes...')
        mixing_params = get_mixing_params(evaltest_mixing_param_file)
        generate_mixture_recipes(data_path=data_path, current_subsets=np.array(['evaltest']), mixing_params=mixing_params)

    if synthesize_devtrain_mixtures:
        mixing_params = get_mixing_params(devtrain_mixing_param_file)
        param_hash = m(yaml.dump(mixing_params).encode('utf-8')).hexdigest()
        subset = 'devtrain'
        if os.path.exists(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio')):
            print('Folder {} already exists. Overwriting.'.format(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio')))
            shutil.rmtree(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio'))

        print('='*30)
        print('Generating {} mixtures...'.format(subset))

        do_mixing(data_path=data_path, current_subsets =np.array([subset]),
                  magic_anticlipping_factor=magic_anticlipping_factor, param_hash=param_hash)

    if synthesize_devtest_mixtures:
        subset = 'devtest'
        mixing_params = get_mixing_params(devtest_mixing_param_file)
        param_hash = m(yaml.dump(mixing_params).encode('utf-8')).hexdigest()
        if os.path.exists(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio')):
            print('Folder {} already exists. Overwriting.'.format(
                os.path.join(data_path, 'mixture_data', subset, param_hash, 'audio')))
            shutil.rmtree(os.path.exists(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio')))
        print('=' * 30)
        print('Generating {} mixtures...'.format(subset))

        do_mixing(data_path=data_path, current_subsets=np.array([subset]),
                  magic_anticlipping_factor=magic_anticlipping_factor, param_hash=param_hash)

    if synthesize_evaltest_mixtures:
        subset = 'evaltest'
        mixing_params = get_mixing_params(evaltest_mixing_param_file)
        param_hash = m(yaml.dump(mixing_params).encode('utf-8')).hexdigest()
        if os.path.exists(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio')):
            print('Folder {} already exists. Overwriting.'.format(
                os.path.join(data_path, 'mixture_data', subset, param_hash, 'audio')))
            shutil.rmtree(os.path.join(data_path, 'mixture_data',subset,param_hash,'audio'))
        print('=' * 30)
        print('Generating {} mixtures...'.format(subset))

        do_mixing(data_path=data_path, current_subsets=np.array([subset]),
                  magic_anticlipping_factor=magic_anticlipping_factor, param_hash=param_hash)


if __name__ == '__main__':
    # parse args

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Rare event detection
            Mixture synthesizer
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Aleksandr Diment ( aleksandr.diment@tut.fi )

            This software performs generation of mixture recipes and synthesis of the mixtures for the DCASE 2017
            Rare sound event detection task. Each mixture consists of at most one rare sound event of the target class.
            The generation uses data provided by the DCASE2017 challenge consisting of backround recordings and target
            isolated sound events. For more details, see http://www.cs.tut.fi/sgn/arg/dcase2017/

        '''))

    parser.add_argument("-data_path", help="Path to data (should include at least source_data "
                                           "folder with forlders bgs, events, cv_setup). "
                                           "Default: {}".format(os.path.join('..', 'data')),
                        default=os.path.join('..', 'data'), dest='data_path')
    parser.add_argument("-subset", help="devtrain, devtest, evaltest, or all",
                        default='devtrain', dest='subset')

    parser.add_argument('-devtrain_params', help='Parameter file (yaml) for the devtrain mixtures '
                                                 '(default: the provided mixing_params.yaml)',
                        required=False, default='mixing_params.yaml', dest='devtrain_params')
    parser.add_argument('-devtest_params', help='Parameter file (yaml) for the devtest mixtures '
                                                 '(default: the provided mixing_params_devtest_dcase_fixed.yaml). '
                                                'Note: keep devtest params default to recreate the fixed DCASE devtest set!',
                        required=False, default='mixing_params_devtest_dcase_fixed.yaml', dest='devtest_params')
    args = parser.parse_args()


    generate_devtrain_recipes = (args.subset == 'devtrain') or (args.subset == 'all')
    synthesize_devtrain_mixtures = generate_devtrain_recipes
    generate_devtest_recipes = (args.subset == 'devtest') or (args.subset == 'all')
    synthesize_devtest_mixtures = generate_devtest_recipes
    generate_evaltest_recipes  = (args.subset == 'evaltest') or (args.subset == 'all')
    synthesize_evaltest_mixtures = generate_evaltest_recipes


    main(data_path=args.data_path,
         generate_devtrain_recipes=generate_devtrain_recipes,
         generate_devtest_recipes=generate_devtest_recipes,
         generate_evaltest_recipes=generate_evaltest_recipes,
         synthesize_devtrain_mixtures=synthesize_devtrain_mixtures,
         synthesize_devtest_mixtures=generate_devtest_recipes,
         synthesize_evaltest_mixtures=generate_evaltest_recipes,
         devtrain_mixing_param_file=args.devtrain_params,
         devtest_mixing_param_file=args.devtest_params)
