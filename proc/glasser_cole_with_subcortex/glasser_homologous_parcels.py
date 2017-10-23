from tools.cifti import Gifti
from tools.io import Data
import numpy as np
from tools.parcels import Parcel

data_input = Data(input_dir='/Users/md2242/Projects/hcp_tools/data/', plot_dir='')


"""
Match Parcels
"""
glasser_lr = Gifti(data_input.input_dir+'templates/glasser_left.label.gii', data_input.input_dir+'templates/glasser_right.label.gii')
parcel_L, parcel_R = glasser_lr.data(0)


left_ordered = np.unique(parcel_L)
right_ordered = np.zeros(len(left_ordered))
for ii, pp in enumerate(left_ordered):
    dummy = parcel_R[parcel_L == pp]
    count = np.zeros(len(np.unique(dummy)))
    for jj, pp2 in enumerate(np.unique(dummy)):
        count[jj] = (dummy == pp2).sum()
    right_ordered[ii] = np.unique(dummy)[count.argmax()]

left_ordered = left_ordered[1:]
right_ordered = right_ordered[1:]

glasser = Parcel()
glasser.drop(na=False, key='surface', value='subcortex')

indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

ixl = glasser.parcel.loc[indices_L].scalar.values
ixr = glasser.parcel.loc[indices_R].scalar.values
order = [np.where(ixr == right_ordered[ii])[0][0] for ii in range(180)]


"""
Double check with original parcel, in which the parcels are homologous
"""
glasser_orig = Gifti(data_input.input_dir+'templates/glasser_orig_left.label.gii', data_input.input_dir+'templates/glasser_orig_right.label.gii')
parcel_L_orig, parcel_R_orig = glasser_orig.data(0)

left_ordered_orig = np.unique(parcel_L_orig)
right_ordered_orig = np.unique(parcel_R_orig)

left2orig = np.zeros(len(left_ordered_orig))
right2orig = np.zeros(len(right_ordered_orig))
for ii, pp in enumerate(left_ordered_orig):
    dummy = parcel_L[parcel_L_orig == pp]
    count = np.zeros(len(np.unique(dummy)))
    for jj, pp2 in enumerate(np.unique(dummy)):
        count[jj] = (dummy == pp2).sum()
    left2orig[ii] = np.unique(dummy)[count.argmax()]

for ii, pp in enumerate(right_ordered_orig):
    dummy = parcel_R[parcel_R_orig == pp]
    count = np.zeros(len(np.unique(dummy)))
    for jj, pp2 in enumerate(np.unique(dummy)):
        count[jj] = (dummy == pp2).sum()
    right2orig[ii] = np.unique(dummy)[count.argmax()]

left2orig = left2orig[1:]
right2orig = right2orig[1:]

import matplotlib.pyplot as plt
plt.subplot(211)
plt.scatter(ixl, left2orig[left2orig.argsort()])
plt.subplot(212)
plt.scatter(ixr[order], right2orig[left2orig.argsort()])
plt.show()


"""
Write results
"""
write = False
if write:
    fout = data_input.save('glasser_homologous_indices.hdf5')
    fout.create_dataset('left_scalar', data=left_ordered)
    fout.create_dataset('right_scalar', data=right_ordered)
    fout.create_dataset('right_reorder', data=order)
    fout.create_dataset('indices_L', data=indices_L)
    fout.create_dataset('indicel_R', data=indices_R[order])
    fout.close()
