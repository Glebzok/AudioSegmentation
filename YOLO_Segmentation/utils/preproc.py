import os
import wave
import yaml
import wavio
from scipy import signal
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from dicttoxml import dicttoxml
from lxml import etree
from jinja2 import Environment, PackageLoader


SAMPLE_RATE = 44100
AUDIO_LEN = 5

def write_meta_yaml(filename, data):
    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(data,default_flow_style=False))
        
def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def generate_yaml_descr_from_txt(path):
    for entry in os.scandir(path):
        if os.path.isdir(entry):
            for subentry in os.scandir(entry):
                if subentry.path[-4:] == '.txt':

                    with open(subentry.path, 'r') as file:
                        onset, offset = str(file.read()).split()
                        onset, offset = float(onset), float(offset)

                    valid_segments = [[onset, offset]]
                    name = subentry.name[:-8]+'.wav'
                    audio_file = wave.open(subentry.path[:-8]+'.wav', 'rb')
                    channels = audio_file.getnchannels()
                    samplerate = audio_file.getframerate()
                    duration = audio_file.getnframes()/samplerate
                    filesize = os.stat(subentry.path[:-8]+'.wav').st_size

                    descr = {'channels': channels,
                            'duration': duration,
                            'filesize': filesize,
                            'samplerate': samplerate,
                            'name': name,
                            'valid_segments': valid_segments}

                    write_meta_yaml(subentry.path[:-8]+'.yaml', descr)
                    
def generate_events_yaml_descr(cv_setup_path, events_path):
    id = 0
    descr = {}
    for entry in os.scandir('events_path'):
        if os.path.isdir(entry) and '.' not in entry.name:
            classname = entry.name
            class_list = []
            for subentry in os.scandir(entry):
                if subentry.path[-5:] == '.yaml':
                    data = read_meta_yaml(subentry.path)
                    for valid_segment in data['valid_segments']:
                        parent_meta = dict(data)
                        del parent_meta['valid_segments']

                        descr_file = {'audio_filename': parent_meta['name'],
                                      'parent_meta': parent_meta,
                                      'segment': valid_segment}
                        id += 1
                        class_list.append(descr_file)

            descr[classname] = class_list

    write_meta_yaml(cv_setup_path+'events_evaltest.yaml', descr)
    
def newest_subdir_of(b):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return max(result, key=os.path.getmtime)
    
def create_spectrogram_from_wav_file(mixture_path, filename, pic_path, channel = 0, plotting = False):
    
    samples = wavio.read(mixture_path+filename).data.reshape((-1,))
    
    frequencies, times, spectrogram = signal.spectrogram(samples, SAMPLE_RATE)
    
    fig=plt.figure(figsize=((13.22, 13.57)))
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.pcolormesh(times, frequencies, np.log(spectrogram), figure = fig)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    name = pic_path+filename[:-3]+'jpg'
    plt.savefig(name, bbox_inches=extent, dpi = 100)
    if plotting:        
        plt.show()
    fig.clear()
    plt.close(fig)
    
    return (name, 1024, 1024)
    
    
class Writer:
    def __init__(self, filename, width, height, depth=3):
        environment = Environment(loader=PackageLoader('pascal_voc_writer'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        self.template_parameters = {
            'filename': filename,
            'width': width,
            'height': height,
            'depth': depth,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)

def create_spectrs_and_anns_from_wav(mixture_path, spectrograms_path, annotations_path):
    with open(mixture_path+'/meta/event_list_evaltest.csv', 'r') as f:
        for j, s in enumerate(f):

            s = s.split()
            filename = s[0]
            pic_name, width, height = create_spectrogram_from_wav_file(mixture_path+'/audio/', filename, spectrograms_path)

            writer = Writer(filename[:-3]+'jpg', width, height, 3)       
            nmb_of_events = int(s[1])

            for i in range(nmb_of_events):
                writer.addObject(s[2 +  3 * i], int(width * float(s[2 +  3 * i + 1]) / AUDIO_LEN), 0, int(width * (float(s[2 +  3 * i + 1]) + float(s[2 +  3 * i + 2])) / AUDIO_LEN), height)

            writer.save(annotations_path+filename[:-3]+'xml')
