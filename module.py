import os
import time
import torch
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image

import spotipy
import spotipy.util as util
from phue import Bridge
from datetime import datetime, date

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python import keras

from pygame import mixer

from cnn import MusicNet


class HueLight:
    # class to control Hue lightbulbs

    def __init__(self):
        super().__init__()
        self.lights = [4, 6, 9] # initialise light IDs for each of the three lights

        # set priority
        self.first = self.lights[0]
        self.second = self.lights[1]
        self.third = self.lights[2]

    def connect_hue(self):

        # connect via bridge and initialise session

        bridge_ip = '' # secret
        self.b = Bridge(bridge_ip) # connect using IP address
        self.b.connect()  # establish connection
        self.hue_connected = True

    def set_lights(self, first, second, third):

        # change the light IDs for each light

        self.first = first
        self.second = second
        self.third = third

    def rotate_lights(self):

        # for a new section, switch the order of lights to provide a different atmosphere

        hold = self.first
        self.first = self.second
        self.second = self.third
        self.third = hold

    def get_bridge_lights(self):

        # get numbers of all lights connected to the hue bridge

        light_ids = self.b.get_light_objects('id')

        for id in light_ids:
            print(id, light_ids[id])

    def get_light_status(self, light):

        # get the status of any light, print all it's relevant characteristics

        light_on = self.b.get_light(light, 'on')
        light_colormode = self.b.get_light(light, 'colormode')
        light_bri = self.b.get_light(light, 'bri')
        light_hue = self.b.get_light(light, 'hue')
        light_sat = self.b.get_light(light, 'sat')
        light_xy = self.b.get_light(light, 'xy')
        light_ct = self.b.get_light(light, 'ct')

        date_ = date.today()
        now = datetime.now()

        current_date = date_.strftime("%d/%m/%y")
        current_time = now.strftime("%H:%M")
        
        print("{} at {}, bulb 4".format(current_date, current_time))
        print("status:", light_on)
        print("colour mode:", light_colormode)
        print("brightness:", light_bri)
        print("hue:", light_hue)
        print("saturation:", light_sat)
        print("xy:", light_xy)
        print("ct:", light_ct)

    def light_on(self, light):

        # set a light to 'on'

        self.b.set_light(light, 'on', True)
        print("light turned on")

    def light_off(self, light):

        # set a light to 'off'

        self.b.set_light(light, 'on', False)
        print("light turned off") 

    def change_one_light(self, light, type, val=None, trans=None):

        # change the value of a single metric for a single light

        if val == None and trans == None: # input is a dictionary
            self.b.set_light(light, type)
        else: # input is not a dictionary
            self.b.set_light(light, type, val, transitiontime=trans)

    def change_lights(self, trans):

        # takes in a dictionary of transitions with key as the light ID

        for i in trans:
            self.b.set_light(i, trans[i])

    def change_brightnesses(self, lights, bri, trans):

        # specifically change the brightness of the light
        
        for light in lights:
            self.b.set_light(light, 'bri', bri, trans)

    def rgb_to_xy(self, rgb):

        # convert an input rgb value to an output xy value
        # conversion taken from https://gist.github.com/popcorn245/30afa0f98eea1c2fd34d

        rgb_norm = [(i/255) for i in rgb]  # normalise each of the red, green, blue values

        gamma_corrected = []

        for i in rgb_norm:
            if i > 0.04045:
                gamma_corrected.append(np.power((i+0.055)/0.055, 2.4))
            else:
                gamma_corrected.append(i/12.92)

        X = (gamma_corrected[0]*0.649926) + (gamma_corrected[1]*0.103455) + (gamma_corrected[2]*0.197109)
        Y = (gamma_corrected[0]*0.234327) + (gamma_corrected[1]*0.743075) + (gamma_corrected[2]*0.022598)
        Z = (gamma_corrected[1]*0.053077) + (gamma_corrected[2]*1.035763)

        xy = [X/(X+Y+Z), Y/(X+Y+Z)]
    
        return xy





class SpotifyLink:

    def __init__(self, uri=None):
        username = '1168538196'
        client_id ='5708cf5e0c1e46d9aec98b0e89891150'
        client_secret = '' # secret
        redirect_uri = 'http://localhost:1410/'

        scope = '' # no account-based access required for session so scope is empty

        token = util.prompt_for_user_token(username=username, 
                                        scope=scope, 
                                        client_id=client_id,   
                                        client_secret=client_secret,     
                                        redirect_uri=redirect_uri)

        self.sp = spotipy.Spotify(auth=token) # initialise session


    def set_new_uri(self, uri):

        # set a new URI (spotify song ID) when a new song is played from input
        self.uri = uri


    def get_track_tempo(self):

        # get the calculated tempo from the feature list associated with the track
        # more reliable method than using Librosa analysis for this

        features = self.sp.audio_features(tracks=self.uri)
        track_features = features[0]

        bpm = float(track_features["tempo"])

        beat_dur = 60/bpm # get length of an individual beat

        while beat_dur < 2.5: # ensure that if beats are quick, lights aren't changing too quickly
                beat_dur *= 2

        self.transition_time = beat_dur # use this multiple of the beat as the transition time


    def get_track_sections(self):

        # get the spotify session analysis for the track

        analysis = self.sp.audio_analysis(self.track_uri)

        sections = []

        for section in analysis['sections']: 
            sections.append(section['start']) # create a list of section start times that can be referenced as the song progresses
        
        return sections






class SongAnalysis(HueLight,SpotifyLink):

    def __init__(self, file, iteration, model, uri, colours):
        super().__init__()

        self.img_height = 235 # initialise image size for analysis by the ML model
        self.img_width = 352

        self.genre_list = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time', 'Pop', 'Rock', 'Soul-RnB', 'Spoken'] # genre list for medium dataset

        # setting up values used to access files and filepaths
        track = file.replace('.mp3', '')
        self.track = track
        self.file = file

        self.parent = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/for_use/'
        self.track_path = os.path.join(self.parent, self.file)
        self.folder_path = os.path.join(self.parent, self.track)
        self.folder_dir = os.path.join(self.folder_path)

        self.iteration = iteration
        self.model_no = model

        # initialise hue connection
        self.connect_hue()
        # load model (both tensorflow and torch models were used at different points)
        if self.iteration == 'torch':
            self.load_torch()
        else:
            self.load_tf()

        # set the uri from the input
        self.uri = uri
        # get track tempo from Spotify data
        self.get_track_tempo()
        # get track sections from Spotify data
        self.sections = self.get_track_sections()
        # set colours from input
        self.set_colours(colours)
        # set lights (unchanged)
        self.set_lights(4,6,9)

        # input genres to control the lighting
        top_1 = input('Enter first genre to track')
        top_2 = input('Enter second genre to track')
        top_3 = input('Enter third genre to track')
        self.set_genres([top_1, top_2, top_3])

        # total duration of track to establish when the end of the track is
        duration = librosa.get_duration(filename=self.track_path)
        self.dur_count = np.ceil(duration/self.transition_time) # how many iterations we go through

    def load_tf(self):

        # load in tensorflow model from folder

        print('loading tf model...')

        self.model = keras.models.load_model('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/iteration_'+self.iteration+'/model_'+self.model_no)

        print('model loaded!')

    def load_torch(self):

        # load in torch model from .pt file

        print('loading torch model...')

        self.model = torch.load('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/torch/model_'+self.model_no+'.pt')

        print('model loaded!')

    def change_track(self, new_file, uri):

        # reset all variables associated with track when a new track is loaded

        track = new_file.replace('.mp3', '')
        self.track = track
        self.file = new_file

        self.track_path = os.path.join(self.parent, self.file)
        self.folder_path = os.path.join(self.parent, self.track)
        self.folder_dir = os.path.join(self.folder_path)

        self.set_new_uri(uri)

        duration = librosa.get_duration(filename=self.track_path)
        self.dur_count = np.ceil(duration/3)

    def set_colours(self, cols):

        # set up colours of each light from rgb input
        
        first, second, third = cols

        self.primary_rgb = first
        self.secondary_rgb = second
        self.tertiary_rgb = third

        # convert to xy colour space

        self.primary_xy = self.rgb_to_xy(self.primary_rgb)
        self.secondary_xy = self.rgb_to_xy(self.secondary_rgb)
        self.tertiary_xy = self.rgb_to_xy(self.tertiary_rgb)

        # set each of the three lights to their colour

        self.change_one_light(self.lights[0], 'xy', self.primary_xy, 1)
        self.change_one_light(self.lights[1], 'xy', self.secondary_xy, 1)
        self.change_one_light(self.lights[2], 'xy', self.tertiary_xy, 1)

    def set_genres(self, genres):

        # set the genres to be tracked wrt the full list of genres, so that the correct indeces can be used when taking values from the score list

        self.genres_chosen = []
        for genre in genres:
            self.genres_chosen.append(self.genre_list.index(genre))

    def xy_test(self, scores):

        # push new brightness as dictated by change in genre score

        for idx, i in enumerate(self.genres_chosen):
            brightness = 127 + scores[i]*127 # unnormalise score with minimum brightness of 127
            self.change_one_light(self.lights[idx], 'bri', brightness, 2) # set for respective light


    def process_segment(self, count):

        # get genre score predictions using ML model

        # load segment data with librosa
        data, sr = librosa.load(self.track_path, res_type='kaiser_fast', offset=count*self.transition_time, duration=self.transition_time)
        
        # create and save a spectrogram for that segment
        spec = librosa.feature.melspectrogram(y=data, sr=sr)
        spec_big = librosa.power_to_db(spec)
        img = librosa.display.specshow(spec_big)
        image_name = self.track + '{:03d}'.format(count)
        img_path = self.parent+"/{}.png".format(image_name)
        plt.savefig(img_path, bbox_inches='tight')
        plt.clf()

        # load the spectrogram in the correct format
        img = tf.keras.utils.load_img(
            img_path, target_size=(self.img_height, self.img_width))

        # convert to an array for processing
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # get predictions from model
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # collect list of scores from the output
        self.scores = []
        for i in score:
            self.scores.append(float(i))
        
        # delete the spectrogram
        os.remove(img_path)

        # return the list of scores
        return self.scores

    def start_song(self):

        # use pygame mixer to play the track
        mixer.init()
        mixer.music.load(self.track_path)
        mixer.music.play()

        # initialise variables for use in checking song duration, section switching, and plotting the genre scores
        count = 0
        section_counter = 1
        self.song_scores = []
        
        while count < self.dur_count: # while song is playing
            start = time.perf_counter() # keep to the time of the music

            # check if section has changed
            if count*self.transition_time+self.transition_time > self.sections[section_counter]:
                self.rotate_lights() # if so, change lights
                section_counter += 1 # change value of reference

            scores = self.process_segment(count) # process the segment
            self.xy_test(scores) # use the scores to change the lights
            count += 1
            remaining = time.perf_counter()-start
            print(remaining)
            if remaining < self.transition_time:
                time.sleep(self.transition_time-remaining) # wait any left over time before moving to next segment

    def process_results(self):

        # plot a graph of the genre scores over time

        df = pd.DataFrame(self.song_scores, columns=self.genre_list)

        df.to_csv('data.csv')
        data = pd.read_csv('data.csv', index_col=False)

        data[self.genre_list].plot()

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.parent+'results/'+self.iteration+'_'+self.model_no+'/'+self.track, bbox_inches='tight')


def initialise_module():

    # get inputs of relevant data for the module

    track_name = input('enter track file name:')
    iteration = input('enter model iteration number:')
    model_no = input('enter model version number:')
    track = track_name+'.mp3'
    model = 'v'+model_no
    uri = input('enter track spotify uri:')

    colour_1 = input('enter the primary rgb colour for {}:'.format(track_name))
    colour_2 = input('enter the secondary rgb colour:')
    colour_3 = input('enter the tertiary rgb colour:')

    colours = [colour_1, colour_2, colour_3]

    # initialise class instance
    analyse = SongAnalysis(track, iteration, model, uri, colours)

    # start song
    play = input('play track? y/n')
    if play == 'y':
        analyse.start_song()
        
    play = input('play another track? y/n')
    if play == 'y':
        initialise_module()
    else:
        pass
    

initialise_module()

mixer.music.stop()