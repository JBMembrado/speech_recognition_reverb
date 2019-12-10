import re
import os
from speech_dataset_load import load_speech_dataset
import reverb_model
import shutil
from datetime import datetime
import random


class ReverbData(object):
    def __init__(self):
        self.raw_dataset = None
        self.raw_only_speech = None
        self.raw_only_noise = None
        self.new_path = None
        self.new_folder_name = None
        self.reverb_model = None
        self.random_params = None

    def load_dataset(self, subset=10):
        """
        Calls the load_speech_dataset function that creates a Dataset object from the GSD.
        """
        print('Loading the Google Speech Dataset.')
        print()
        self.raw_dataset = load_speech_dataset(subset)
        print('Google Speech Dataset successfully loaded.')
        print()

    def separate_dataset(self):
        """
        Separates the dataset into speech/noise.
        """
        print('Separate dataset between speech and noise :')
        self.raw_only_speech = self.raw_dataset.filter(speech=True)
        self.raw_only_noise = self.raw_dataset.filter(speech=False)
        print('Successfully separated the dataset.')

    def init_new_folder(self, new_name):
        """
        This method creates and initiates the subfolders containing each wav file.
        """
        self.new_folder_name = new_name
        self.new_path = re.sub('data', self.new_folder_name, self.raw_dataset.basedir)

        print('The new folder is located at : {0}'.format(self.new_path))
        if not os.path.isdir(self.new_path):
            os.makedirs(self.new_path)

        for subdir in self.raw_dataset.subdirs:
            extracted_subdir = re.split('/', subdir)[-2]
            # We do not want to apply reverb on the background noise folder,
            # so we exclude this folder during this step
            if extracted_subdir != '_background_noise_':
                if not os.path.exists(self.new_path + '/' + extracted_subdir):
                    os.makedirs(self.new_path + '/' + extracted_subdir)

        print('New Folder initiated.')

    def init_reverb_model(self):
        self.reverb_model = reverb_model.ReverbModel()
        self.reverb_model.reverb_model3()

    def init_random_reverb_model(self):
        params = random.choice(self.random_params)

        self.reverb_model = reverb_model.ReverbModel()
        self.reverb_model.reverb_model_generic(params[0], params[1], params[2])

    def apply_reverb(self):
        print('Applying reverberation algorithm to the whole dataset : ')

        for idx_sample, sample in enumerate(self.raw_only_speech):
            self.init_reverb_model()
            audio_sample_no_rev = sample.data
            output_path = re.sub('data', self.new_folder_name, sample.meta.file_loc)
            self.reverb_model.transform_audio(audio_sample_no_rev, output_path)

            if (idx_sample % 500 == 0) and (idx_sample > 0):
                print('Now treating sample {0}.'.format(idx_sample))

        print('Reverberation applied to the whole speech dataset.')

    def apply_reverb_random(self, set_params):
        print('Applying reverberation to the whole dataset with randomized parameters.')
        self.random_params = set_params

        for idx_sample, sample in enumerate(self.raw_only_speech):
            self.init_random_reverb_model()
            audio_sample_no_rev = sample.data
            output_path = re.sub('data', self.new_folder_name, sample.meta.file_loc)
            self.reverb_model.transform_audio(audio_sample_no_rev, output_path)

            if (idx_sample % 500 == 0) and (idx_sample > 0):
                print('Now treating sample {0}.'.format(idx_sample))

        print('Reverberation applied to the whole speech dataset.')

    def copy_files(self):
        testing_list = self.raw_dataset.basedir + '/testing_list.txt'
        validation_list = self.raw_dataset.basedir + '/validation_list.txt'

        shutil.copy2(testing_list, re.sub('data', self.new_folder_name, testing_list))
        shutil.copy2(validation_list, re.sub('data', self.new_folder_name, validation_list))

        noise_folder = self.raw_dataset.basedir + '/_background_noise_/'
        shutil.copytree(noise_folder, re.sub('data', self.new_folder_name, noise_folder))

        print('Extra files copied.')


start = datetime.now()

# params = [room_dimensions, source_position, mic_position]
params1 = [[6, 5, 7], [2, 3.1, 2], [2, 1.5, 2]]
params2 = [[3, 2, 4], [1, 1, 1], [2, 1, 2]]
params3 = [[6, 5, 7], [2, 3.1, 2], [5, 0.5, 5]]
params4 = [[7, 6, 7], [1, 2, 2], [6, 2, 6]]

set_params = [params1, params2, params3, params4]

rev_data = ReverbData()
rev_data.load_dataset(subset=5)
rev_data.separate_dataset()
rev_data.init_new_folder('mixed_params_reverb')
rev_data.apply_reverb_random(set_params)
rev_data.copy_files()

time_elapsed = datetime.now() - start
print('Time elapsed (hh:mm:ss.ms) {0}'.format(time_elapsed))
