import numpy as np
import pandas as pd
import nibabel as nib
import h5py

def process_noise():
	# folder structure
	syn5_folder = ''

	Noise_file = h5py.File(syn5_folder+'noise.hdf5','r')
	subjects = Noise_file['subjects'].value
	N_subjects = len(subjects)
	subjects_string = [str(subjects[ii]) for ii in range(N_subjects)]
	
	f = h5py.File('noise_processed.hdf5','w')
	f.create_dataset("subjects", data=subjects, compression="lzf")

	t_ds = 288#400*0.72
	grp = []
	# Read from file to array
	for ii in range(N_subjects):
		grp += [f.create_group(subjects_string[ii])]
		for jj in range(4):
			pulse_raw = Noise_file[subjects_string[ii]]['pulse_'+str(jj+1)]
			resp_raw = Noise_file[subjects_string[ii]]['respiration_'+str(jj+1)]
			hr_raw = Noise_file[subjects_string[ii]]['heartrate_'+str(jj+1)]
					
			pulse_av, pulse_var = np.zeros(1200), np.zeros(1200)
			resp_av, resp_var = np.zeros(1200), np.zeros(1200)
			hr_av, hr_var = np.zeros(1200), np.zeros(1200)
			
			pulse_av = np.array([pulse_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
			resp_av = np.array([resp_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
			hr_av = np.array([hr_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
			
			pulse_var = np.array([pulse_raw[t_ds*kk:t_ds*kk+t_ds].var() for kk in range(1200)])
			resp_var = np.array([resp_raw[t_ds*kk:t_ds*kk+t_ds].var() for kk in range(1200)])
			hr_var = np.array([hr_raw[t_ds*kk:t_ds*kk+t_ds].var() for kk in range(1200)])
	
			grp[-1].create_dataset("pulse_av_" + str(jj+1), data=pulse_av, compression="lzf")
			grp[-1].create_dataset("respiration_av_" + str(jj+1), data=resp_av, compression="lzf")
			grp[-1].create_dataset("heartrate_av_" + str(jj+1), data=hr_av, compression="lzf")

			grp[-1].create_dataset("pulse_var_" + str(jj+1), data=pulse_var, compression="lzf")
			grp[-1].create_dataset("respiration_var_" + str(jj+1), data=resp_var, compression="lzf")
			grp[-1].create_dataset("heartrate_var_" + str(jj+1), data=hr_var, compression="lzf")
		
			movement_absolute = Noise_file[subjects_string[ii]]['AbsoluteMovement_'+str(jj+1)]
			movement_relative = Noise_file[subjects_string[ii]]['RelativeMovement_'+str(jj+1)]
			
			grp[-1].create_dataset("AbsoluteMovement_" + str(jj+1), data=movement_absolute, compression="lzf")
			grp[-1].create_dataset("RelativeMovement_" + str(jj+1), data=movement_relative, compression="lzf")
			
		print ii

process_noise()