import numpy as np
from tools.io import Data

"""
Collect Structural measures
"""
def get_structural():
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_demean_4_sessions.hdf5')
    sc = file['sc'].value.mean(2)
    file.close()

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    psize = glasser.parcel.parcel_size.values
    psize_mat = np.tile(psize, (360, 1))
    psize_mat = psize_mat * psize_mat.T

    sc_diag = np.copy(np.diag(sc))
    for ii in range(360): sc[ii, ii] = 0.0

    fout = data_input.save('hcp_334_cortex_average_structural.hdf5')
    fout.create_dataset('sc', data=sc, compression="lzf")
    fout.create_dataset('sc_diag', data=sc_diag, compression="lzf")
    fout.create_dataset('psize', data=psize, compression="lzf")
    fout.create_dataset('psize_mat', data=psize_mat, compression="lzf")
    fout.create_dataset('indices_L', data=indices_L, compression="lzf")
    fout.create_dataset('indices_R', data=indices_R, compression="lzf")
    fout.close()


def get_myelin():
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_demean_4_sessions.hdf5')
    myelin_raw = file['myelin'].value
    thickness = file['thickness'].value
    file.close()

    file = data_input.load('glasser_cole/hcp_334_structural_final.hdf5')
    myelin_bc = file['myelin'].value
    file.close()

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')
    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    file_homologous = data_input.load('glasser_cole/glasser_homologous_indices.hdf5')
    order = file_homologous['right_reorder'].value
    file_homologous.close()

    fin = data_input.load('glasser_cole/conte69_myelin.hdf5')
    myelin_conte = fin['myelin'].value
    fin.close()

    fin = data_input.load('glasser_cole/surrogate_myelin_maps.hdf5')
    surrogate_maps = fin['surrogates'].value
    rho = fin['rho'].value
    d0 = fin['d0'].value
    distance_matrix = fin['distance_matrix'].value
    fin.close()

    print surrogate_maps.shape
    surrogate_maps_full = np.zeros((surrogate_maps.shape[0], 360))
    for ii in range(surrogate_maps.shape[0]):
        surrogate_maps_full[ii,indices_L] = surrogate_maps[ii]
        surrogate_maps_full[ii,indices_R[order]] = surrogate_maps[ii]

    #fout = data_input.save('hcp_334_cortex_surrogate_myelin_maps.hdf5')
    #fout.create_dataset('surrogate', data=surrogate_maps_full, compression="lzf")
    #fout.close()
    #import pdb; pdb.set_trace()


    myelin_bc_sym_indv = np.zeros((360, 334))
    for ii in range(334):
        myelin_bc_sym_indv[indices_L, ii] = (myelin_bc[indices_L, ii] + myelin_bc[indices_R, ii][order]) * 0.5
        myelin_bc_sym_indv[indices_R[order], ii] = (myelin_bc[indices_L, ii] + myelin_bc[indices_R, ii][order]) * 0.5

    myelin_raw = myelin_raw.mean(1)

    conte_sym = np.zeros(360)
    conte_sym[indices_L] = (myelin_conte[indices_L] + myelin_conte[indices_R][order]) * 0.5
    conte_sym[indices_R[order]] = (myelin_conte[indices_L] + myelin_conte[indices_R][order]) * 0.5


    myelin_bc_mean = myelin_bc.mean(1)
    myelin_bc_median = np.median(myelin_bc, 1)

    myelin_bc_mean_sym = np.zeros(360)
    myelin_bc_mean_sym[indices_L] = (myelin_bc_mean[indices_L] + myelin_bc_mean[indices_R][order]) * 0.5
    myelin_bc_mean_sym[indices_R[order]] = (myelin_bc_mean[indices_L] + myelin_bc_mean[indices_R][order]) * 0.5

    myelin_bc_median_sym = np.zeros(360)
    myelin_bc_median_sym[indices_L] = (myelin_bc_median[indices_L] + myelin_bc_median[indices_R][order]) * 0.5
    myelin_bc_median_sym[indices_R[order]] = (myelin_bc_median[indices_L] + myelin_bc_median[indices_R][order]) * 0.5

    av_thickness = thickness.mean(1)

    fout = data_input.save('hcp_334_cortex_myelin_maps.hdf5')
    fout.create_dataset('thickness', data=av_thickness, compression="lzf")
    fout.create_dataset('myelin_sym_indv', data=myelin_bc_sym_indv, compression="lzf")
    fout.create_dataset('myelin_raw', data=myelin_raw, compression="lzf")
    fout.create_dataset('myelin_conte', data=myelin_conte, compression="lzf")
    fout.create_dataset('myelin_contesym', data=conte_sym, compression="lzf")
    fout.create_dataset('myelin_bcmean', data=myelin_bc_mean, compression="lzf")
    fout.create_dataset('myelin_bcmedian', data=myelin_bc_median, compression="lzf")
    fout.create_dataset('myelin_bcmeansym', data=myelin_bc_mean_sym, compression="lzf")
    fout.create_dataset('myelin_bcmediansym', data=myelin_bc_median_sym, compression="lzf")
    fout.create_dataset('rho', data=rho)
    fout.create_dataset('d0', data=d0)
    fout.create_dataset('distance_matrix', data=distance_matrix)
    for ii in range(surrogate_maps.shape[0]):
        fout.create_dataset('surrogate_'+str(int(ii+1)), data=surrogate_maps_full[ii], compression="lzf")

    fout.create_dataset('order', data=order, compression="lzf")
    fout.close()





def test_myelin():
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('hcp_334_cortex_myelin_maps.hdf5', from_path=out_dir)
    toplot = file['myelin_bcmedian'].value
    file.close()

    from plot.scifig import Figure
    fig = Figure(size=(2.2, 1.5), dpi=100)
    fig.subplot(11, margins=(0.01, 0.01), scale=0.97)
    thickness_av = np.hstack((np.zeros(19), toplot))
    fig[0].set_brain(cifti_extension='dlabel', cifti_template='glasser', dpi=300)
    fig[0].brain.plot(thickness_av, cmap='OrRd', hemi_label=False, border=False, title='')
    fig[0].brain_cbar(cbar_loc='bottom', append=True, cbar_title=None)
    fig.show()#panel.savefig('myelin_bc_psym.png', dpi=300)

data_dir = '/Users/md2242/Projects/hcp_tools/data/'
out_dir = '/Users/md2242/Projects/hcp_tools/output/forHeterogeneity/'

#get_structural()
#get_myelin()
#test_myelin()

data_input = Data(input_dir=data_dir, output_dir=out_dir)
file = data_input.load('glasser_cole/raw/DWI_WholeBrain.hdf5')
conn1 = np.zeros((360,360,334))
conn3 = np.zeros((360,360,334))

for ii,ss in enumerate(file['subjects'].value):
    conn1[:,:,ii] = file[str(ss)]['conn_1'].value[19:,19:]
    conn3[:, :, ii] = file[str(ss)]['conn_3'].value[19:,19:]

file.close()

file = data_input.save('structural_connectivity.hdf5')
file.create_dataset('conn_1', data=conn1, compression='lzf')
file.create_dataset('conn_3', data=conn3, compression='lzf')
file.close()

import pdb; pdb.set_trace()