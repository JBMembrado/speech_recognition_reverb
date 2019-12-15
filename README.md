# Keyword recognition

This is the project for the COM-415 course.
It contains two parts : 
- keyword recognition in reverberated environment
- realtime keyword recognition

# Reverberated keyword recognition
This part is represented by the files :
- reverb_data.py : contains the ReverbData class, and can be run as it is. It transforms the downloaded dataset into a reverberated dataset. Model can be changed in the code.
- reverb_model.py : contains the ReverbModel class that is called by reverb_data.py to process audio through a reverberation algorithm.
- speech_dataset_load.py : contains a method to load the Google Speech Dataset. It is called by reverb_data.py.
- test_model.py : contains the TestModel class. It is used to test the trained graph and store the results in a numpy array.

# Realtime keyword recognition

This part is represented by the folder real_time, inside which we can find:
- real_time.py : file to run in order to see our real time classification
- label_wav_realtime.py : runs the tensorflow graph in order to make predictions.
- conv_labels.txt : loaded by label_wav_realtime to have the labels
-first_model_graph_anechoic.pb : the tensorflow graph weights that were first trained
