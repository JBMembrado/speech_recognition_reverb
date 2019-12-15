from __future__ import print_function
import numpy as np
import pyroomacoustics as pra


class ReverbModel(object):
    def __init__(self):
        self.room_dimensions = None
        self.shoebox = None
        self.absorption = None
        self.fs = None
        self.max_order = None
        self.mic = None
        self.source_position = None

    def reverb_model1(self, absorption = 0.2, max_order = 15):
        self.room_dimensions = [6, 5, 7]
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = [2, 3.1, 2]
        self.mic = pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, self.shoebox.fs)

    def reverb_model2(self, absorption = 0.1, max_order = 15):
        self.room_dimensions = [6, 5, 7]
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = [2, 3.1, 2]
        self.mic = pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, self.shoebox.fs)

    def reverb_model3(self, absorption = 0.2, max_order = 15):
        self.room_dimensions = [8, 7, 9]
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = [2, 3.1, 2]
        self.mic = pra.MicrophoneArray(np.array([[6, 1.5, 6]]).T, self.shoebox.fs)

    def reverb_model4(self, absorption = 0.2, max_order = 15):
        self.room_dimensions = [6, 5, 7]
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = [2, 3.1, 2]
        self.mic = pra.MicrophoneArray(np.array([[5, 0.5, 5]]).T, self.shoebox.fs)

    def no_reverb(self):
        self.room_dimensions = [2, 2, 2]
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = [2, 3.1, 2]
        self.mic = pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, self.shoebox.fs)

    def reverb_model_generic(self, room_dimensions, source_position, mic_position, absorption = 0.15, max_order = 15):
        self.room_dimensions = room_dimensions
        self.absorption = absorption
        self.max_order = max_order
        self.fs = 16000

        self.shoebox = pra.ShoeBox(
            self.room_dimensions,
            absorption=self.absorption,
            fs=self.fs,
            max_order=self.max_order,
        )

        self.source_position = source_position
        self.mic = pra.MicrophoneArray(np.array([mic_position]).T, self.shoebox.fs)

    def transform_audio(self, audio_sample, output_path):
        """
        Applies reverberation to the audio_sample.
        The source in the virtual room plays the audio_sample signal.
        The output is the result of the captation of the microphone in the virtual room.
        :param audio_sample: data signal we want to apply reverb on
        :param output_path: path where we want to store the output of the reverberation algorithm
        """
        self.shoebox.add_source(self.source_position, signal=audio_sample)
        self.shoebox.add_microphone_array(self.mic)

        self.shoebox.simulate()
        self.shoebox.mic_array.to_wav(output_path, norm=True, bitdepth=np.int16)
