import numpy as np
from scipy import io
from preprocessing import *
import pickle

data=io.loadmat('datasets/m1_data_raw.mat')    
spike_times=data['spike_times'] #Load spike times of all neurons
vels=data['vels'] #Load x and y velocities
vel_times=data['vel_times'] #Load times at which velocities were recorded  
t_start=vel_times[0] #Time to start extracting data - here the first time velocity was recorded
t_end=vel_times[-1]
downsample_factor=1

#When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
#First, we will put spike_times in a cleaner format: an array of arrays
spike_times=np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]=np.squeeze(spike_times[i])

dt=.05 #Size of time bins (in seconds)

#### FORMAT OUTPUT ####
#Bin output (velocity) data using "bin_output" function
vels_binned=bin_output(vels, vel_times, dt, t_start, t_end, downsample_factor)

#### FORMAT INPUT ####
#Bin neural data using "bin_spikes" function
neural_data=bin_spikes(spike_times, dt, t_start, t_end)

data_folder='datasets/' 

with open(data_folder+'neural_data.pickle','wb') as f:
    pickle.dump([neural_data, vels_binned],f)

