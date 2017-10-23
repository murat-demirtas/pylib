import numpy as np
import pandas as pd
import nibabel as nib
import h5py

def write_pscalars():
    # Load demographics
	demographics = pd.read_excel('hcp_demographics.xlsx')
	n334 = demographics['Subject'][demographics['N334']].values

	f = h5py.File('parcellated_variance.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")
	
	grp = []
	# Read from file to array
	for ii in n334:
		subject = str(ii)
		grp += [f.create_group(subject)]
		scans = ['rfMRI_REST1_RL','rfMRI_REST1_LR','rfMRI_REST2_RL','rfMRI_REST2_LR']
		session = ['rest_1','rest_2','rest_3','rest_4']
		for jj, scan in enumerate(scans):
		    of = nib.load('variance/' + subject + '_' + scan + '_var.pscalar.nii')
		    data = np.array(np.squeeze(of.get_data())).T
		    grp[-1].create_dataset(session[jj], data=data, compression="lzf")
		    
	f.close()
		    
		
def write_BOLD_7T():
	import os.path
	# folder structure
	#syn5_folder = '/Volumes/syn5/Studies/Connectome/Parcellated/BOLD_MSMAll_hp2000/'
	#syn10_folder = '/Volumes/syn10/Studies/Connectome/HCP_Modelling/GlasserCole/'
	syn5_folder = 'parcellated/'
	rest = ['rfMRI_REST1_7T_PA', 'rfMRI_REST2_7T_AP', 'rfMRI_REST3_7T_PA', 'rfMRI_REST4_7T_AP']
	movie = ['tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP']
	file_abbr = '_Atlas_MSMAll_hp2000_clean.ptseries.nii'

	# Load demographics
	#demographics = pd.read_excel('hcp_demographics.xlsx')
	#n334 = demographics['Subject'][demographics['7T']].values
	#import pdb; pdb.set_trace()
	n334 = np.loadtxt('HCP_7T_subject.txt')
	f = h5py.File('BOLD_7T_MSMAll_hp2000_clean.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")
	grp = []
	# Read from file to array
	for ii in n334:
			subject = str(int(ii))
			grp += [f.create_group(subject)]
			for jj, rr in enumerate(rest):
				fname = syn5_folder + 'resting/' + subject + '.GCP.' + rr + file_abbr
				if os.path.isfile(fname):
					of = nib.load(fname)
					data = np.array(np.squeeze(of.get_data())).T
					grp[-1].create_dataset("rest_"+str(jj+1), data=data, compression="lzf")
				else:
				    grp[-1].create_dataset("rest_"+str(jj+1), data=np.empty(1))
				
			for jj, rr in enumerate(movie):
				fname = syn5_folder + 'movie/' + subject + '.GCP.' + rr + file_abbr
				if os.path.isfile(fname):
					of = nib.load(fname)
					data = np.array(np.squeeze(of.get_data())).T
					grp[-1].create_dataset("movie_"+str(jj+1), data=data, compression="lzf")
				else:
				    grp[-1].create_dataset("movie_"+str(jj+1), data=np.empty(1))

	f.close()

def write_BOLD():
	# folder structure
	#syn5_folder = '/Volumes/syn5/Studies/Connectome/Parcellated/BOLD_MSMAll_hp2000/'
	#syn10_folder = '/Volumes/syn10/Studies/Connectome/HCP_Modelling/GlasserCole/'
	syn5_folder = 'GlasserCole_raw/'
	file_abbr = '_bold1_4_Atlas_MSMAll_hp2000_clean_LR_Colelab_partitions_v1d_islands_withsubcortex.ptseries.nii'

	# Load demographics
	demographics = pd.read_excel('hcp_demographics.xlsx')
	n334 = demographics['Subject'][demographics['N334']].values

	f = h5py.File('BOLD_MSMAll_hp2000_clean.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")

	grp = []
	# Read from file to array
	for ii in n334:
		subject = str(ii)
		grp += [f.create_group(subject)]
	
		of = nib.load(syn5_folder + subject + file_abbr)
		data = np.array(np.squeeze(of.get_data())).T
		rest1 = data[:,:1200]
		rest2 = data[:,1200:2400]
		rest3 = data[:,2400:3600]
		rest4 = data[:,3600:]

		grp[-1].create_dataset("rest_1", data=rest1, compression="lzf")
		grp[-1].create_dataset("rest_2", data=rest2, compression="lzf")
		grp[-1].create_dataset("rest_3", data=rest3, compression="lzf")
		grp[-1].create_dataset("rest_4", data=rest4, compression="lzf")

	f.close()


def write_DWI():
	# folder structure
	#syn5_folder = '/Volumes/syn5/Studies/Connectome/Parcellated/BOLD_MSMAll_hp2000/'
	#syn10_folder = '/Volumes/syn10/Studies/Connectome/HCP_Modelling/GlasserCole/'
	syn5_folder = 'dwi/'
	file_abbr = '_LR_Colelab_partitions_v1d_islands_withsubcortex.pconn.nii'

	# Load demographics
	demographics = pd.read_excel('hcp_demographics.xlsx')
	n334 = demographics['Subject'][demographics['N334']].values
	
	f = h5py.File('DWI_WholeBrain.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")

	grp = []
	# Read from file to array
	for ii in n334:
		subject = str(ii)
		grp += [f.create_group(subject)]
	
		of1 = nib.load(syn5_folder + subject + '_Conn1' + file_abbr)
		dwi_1 = np.array(np.squeeze(of1.get_data()))
		of3 = nib.load(syn5_folder + subject + '_Conn3' + file_abbr)
		dwi_3 = np.array(np.squeeze(of3.get_data()))

		grp[-1].create_dataset("conn_1", data=dwi_1, compression="lzf")
		grp[-1].create_dataset("conn_3", data=dwi_3, compression="lzf")

	f.close()

def write_Structural():
	# folder structure
	#syn5_folder = '/Volumes/syn5/Studies/Connectome/Parcellated/BOLD_MSMAll_hp2000/'
	#syn10_folder = '/Volumes/syn10/Studies/Connectome/HCP_Modelling/GlasserCole/'
	syn5_folder = 'supplementary/supplementary/'
	file_abbr = '_MSMAll.32k_fs_LR.pscalar.nii'

	# Load demographics
	demographics = pd.read_excel('hcp_demographics.xlsx')
	n334 = demographics['Subject'][demographics['N334']].values

	f = h5py.File('Structure_MSMAll.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")

	grp = []
	# Read from file to array
	for ii in n334:
		subject = str(ii)
		grp += [f.create_group(subject)]
	
		of_myelin = nib.load(syn5_folder + subject + '.MyelinMap' + file_abbr)
		myelin = np.array(np.squeeze(of_myelin.get_data()))
		
		of_thickness = nib.load(syn5_folder + subject + '.corrThickness' + file_abbr)
		thickness = np.array(np.squeeze(of_thickness.get_data()))

		of_sulcus = nib.load(syn5_folder + subject + '.sulc' + file_abbr)
		sulcus = np.array(np.squeeze(of_sulcus.get_data()))

		of_adist = nib.load(syn5_folder + subject + '.ArealDistortion' + file_abbr)
		adist = np.array(np.squeeze(of_adist.get_data()))

		of_edist = nib.load(syn5_folder + subject + '.EdgeDistortion' + file_abbr)
		edist = np.array(np.squeeze(of_edist.get_data()))

		grp[-1].create_dataset("myelin_map", data=myelin, compression="lzf")
		grp[-1].create_dataset("thickness", data=thickness, compression="lzf")
		grp[-1].create_dataset("sulcus", data=sulcus, compression="lzf")
		grp[-1].create_dataset("areal_distortion", data=adist, compression="lzf")
		grp[-1].create_dataset("edge_distortion", data=edist, compression="lzf")

	f.close()

'''
not yet implemented
'''
def write_noise():
	# folder structure
	#syn5_folder = '/Volumes/syn5/Studies/Connectome/Parcellated/BOLD_MSMAll_hp2000/'
	#syn10_folder = '/Volumes/syn10/Studies/Connectome/HCP_Modelling/GlasserCole/'
	syn5_folder = 'supplementary/'

	# Load demographics
	demographics = pd.read_excel('hcp_demographics.xlsx')
	n334 = demographics['Subject'][demographics['N334']].values

	f = h5py.File('noise.hdf5','w')
	f.create_dataset("subjects", data=n334, compression="lzf")
	
	t_ds = 288#400*0.72
	grp = []
	# Read from file to array
	for ii in n334:
		subject = str(ii)
		grp += [f.create_group(subject)]
		scans = ['rfMRI_REST1_RL','rfMRI_REST1_LR','rfMRI_REST2_RL','rfMRI_REST2_LR']
		for jj, scan in enumerate(scans):
			if scan == 'rfMRI_REST2_RL':
				physio = np.loadtxt(syn5_folder + subject + '/' + scan + '/' + scan + '_LR_Physio_log.txt')
			else:
				physio = np.loadtxt(syn5_folder + subject + '/' + scan + '/' + scan + '_Physio_log.txt')
			pulse = physio[:,0]
			resp = physio[:,1]
			hr = physio[:,2]
	
			#pulse = np.zeros(1200)
			#resp = np.zeros(1200)
			#r = np.zeros(1200)
			#pulse = np.array([pulse_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
			#resp = np.array([resp_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
			#hr = np.array([hr_raw[t_ds*kk:t_ds*kk+t_ds].mean() for kk in range(1200)])
	
			grp[-1].create_dataset("pulse_" + str(jj+1), data=pulse, compression="lzf")
			grp[-1].create_dataset("respiration_" + str(jj+1), data=resp, compression="lzf")
			grp[-1].create_dataset("heartrate_" + str(jj+1), data=hr, compression="lzf")
			
			movement_absolute = np.loadtxt(syn5_folder + subject + '/' + scan + '/' + 'Movement_AbsoluteRMS.txt')
			movement_relative = np.loadtxt(syn5_folder + subject + '/' + scan + '/' + 'Movement_RelativeRMS.txt')
			
			grp[-1].create_dataset("AbsoluteMovement_" + str(jj+1), data=movement_absolute, compression="lzf")
			grp[-1].create_dataset("RelativeMovement_" + str(jj+1), data=movement_relative, compression="lzf")
			
		print ii
			#import pdb; pdb.set_trace()


#write_DWI()
#write_BOLD()
#write_Structural()
#write_noise()
#write_BOLD_7T()
write_pscalars()
