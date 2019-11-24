import re
import os
from speech_dataset_load import load_speech_dataset
import reverb_model
import shutil
from datetime import datetime

class ReverbData(object):
    def __init__(self):
        self.raw_dataset = None
        self.raw_only_speech = None
        self.raw_only_noise = None
        self.new_path = None
        self.new_folder_name = None
        self.reverb_model = None

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

    def init_new_folder(self):
        """
        This method creates and initiates the subfolders containing each wav file.
        """
        self.new_folder_name = 'data_reverb'
        self.new_path = re.sub('data', self.new_folder_name, self.raw_dataset.basedir)

        print('The new folder is located at : {0}'.format(self.new_path))
        if not os.path.isdir(self.new_path):
            os.makedirs(self.new_path)

        for subdir in self.raw_dataset.subdirs:
            extracted_subdir = re.split('/', subdir)[-2]
            if not os.path.exists(self.new_path + '/' + extracted_subdir):
                os.makedirs(self.new_path + '/' + extracted_subdir)

        print('New Folder initiated.')

    def init_reverb_model(self):
        self.reverb_model = reverb_model.ReverbModel()
        # example_audio_sample = self.raw_only_speech[0].meta.file_loc
        self.reverb_model.reverb_model1()

    def apply_reverb(self):
        print('Applying reverberation algorithm to the whole dataset : ')

        for idx_sample, sample in enumerate(self.raw_only_speech):
            self.init_reverb_model()
            audio_sample_no_rev = sample.data
            output_path = re.sub('data', self.new_folder_name, sample.meta.file_loc)
            self.reverb_model.transform_audio(audio_sample_no_rev, output_path)

            if (idx_sample%500 == 0) and (idx_sample > 0):
                print('Now treating sample {0}.'.format(idx_sample))

        print('Reverberation applied to the whole speech dataset.')

    def copy_files(self):
        testing_list = self.raw_dataset.basedir + '/testing_list.txt'
        validation_list = self.raw_dataset.basedir + '/validation_list.txt'

        shutil.copy2(testing_list, re.sub('data', self.new_folder_name, testing_list))
        shutil.copy2(validation_list, re.sub('data', self.new_folder_name, validation_list))
        print('Extra files copied.')


start = datetime.now()

rev_data = ReverbData()
rev_data.load_dataset(subset=None)
rev_data.separate_dataset()
rev_data.init_new_folder()
rev_data.apply_reverb()
rev_data.copy_files()

time_elapsed = datetime.now() - start
print('Time elapsed (hh:mm:ss.ms) {0}'.format(time_elapsed))
