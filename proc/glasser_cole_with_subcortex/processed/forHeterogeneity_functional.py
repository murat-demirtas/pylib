import numpy as np
from tools.io import Data
from tools.timeseries import TimeSeries
from tools.linalg import subdiag
from scipy.stats import spearmanr
from tools import utils

def normalize(cov_matrix, av_variance):
    sum_cov = cov_matrix.sum()
    norm_av_var = av_variance / av_variance.sum()
    norm_cov_matrix = cov_matrix / sum_cov
    return norm_cov_matrix, norm_av_var

def normalize_cov(cov_matrix):
    return cov_matrix / cov_matrix.sum()

def normalize_spectrum(psd):
    N = psd.shape[0]
    for ii in range(N):
        psd[ii,:] = psd[ii,:] / psd[ii,:].sum()
    return psd


def indv_variability(fc, measure='rank'):
    N_regions = fc.shape[0]
    N_subjects = fc.shape[2]
    fc_flat = np.empty((N_regions, N_regions-1, N_subjects))
    for s in xrange(N_subjects):
        for n in xrange(N_regions):
            index = np.concatenate((np.arange(n), np.arange(n + 1, N_regions)))
            fc_flat[n, :, s] = fc[n, index, s]
    if measure =='rank':
        if N_subjects < 3:
            fmv2 = np.array([spearmanr(fc_flat[ii])[0] for ii in xrange(N_regions)])
        else:
            fmv2 = np.array([subdiag(spearmanr(fc_flat[ii])[0]) for ii in xrange(N_regions)])
    else:
        fmv2 = np.array([subdiag(np.corrcoef(fc_flat[ii].T)) for ii in xrange(N_regions)])
    return fmv2

def iiv_old(fc):
    N_regions = fc.shape[0]
    N_subjects = fc.shape[2]
    fmv = []
    for p1 in xrange(N_subjects):
        print p1
        mvi = []
        for p2 in xrange(N_subjects):
            if p2 != p1:
                mvi.append([spearmanr(fc[k, np.concatenate((np.arange(k), np.arange(k + 1, N_regions))), p1],
                                      fc[k, np.concatenate((np.arange(k), np.arange(k + 1, N_regions))), p2])[0] for k in xrange(N_regions)])
        fmv.append(mvi)
    return np.array(fmv)

def get_full(x,y):
    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    out = np.zeros(360)
    out[indices_L] = x
    out[indices_R] = y
    return out

"""
Collect rSNR maps
"""
def get_tsnr():
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    sc_file = data_input.load('glasser_cole/raw/DWI_WholeBrain.hdf5')
    fc_file = data_input.load('glasser_cole/raw/BOLD_MSMAll_hp2000_clean.hdf5')
    subjects = utils.hcp_subjects(sc_file)
    N_subjects = len(subjects)
    tsnr = np.empty((360, 4, N_subjects))
    for ii, ss in enumerate(subjects):
        ts = TimeSeries(utils.hcp_get(fc_file, ss, ['rest_1']), fs=0.72, standardize=False, demean=False, gsr=False,
                        index=19)
        tsnr[:, 0, ii] = ts.ts.mean(1) / ts.ts.std(1)

        ts = TimeSeries(utils.hcp_get(fc_file, ss, ['rest_2']), fs=0.72, standardize=False, demean=False, gsr=False,
                        index=19)
        tsnr[:, 1, ii] = ts.ts.mean(1) / ts.ts.std(1)

        ts = TimeSeries(utils.hcp_get(fc_file, ss, ['rest_3']), fs=0.72, standardize=False, demean=False, gsr=False,
                        index=19)
        tsnr[:, 2, ii] = ts.ts.mean(1) / ts.ts.std(1)


        ts = TimeSeries(utils.hcp_get(fc_file, ss, ['rest_4']), fs=0.72, standardize=False, demean=False, gsr=False,
                        index=19)
        tsnr[:, 3, ii] = ts.ts.mean(1) / ts.ts.std(1)

    return tsnr


"""
Collect empirical FC matrices
"""
def get_empirical(N, ns):
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_zscored_single_session.hdf5')
    fc_all = file['fc'].value
    file.close()

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    fc_l = np.empty((180,180,N))
    fc_r = np.empty((180, 180, N))
    for ii in range(N):
        dummy = fc_all[:,:,ii]
        fc_l[:,:,ii] = dummy[indices_L, :][:, indices_L]
        fc_r[:,:,ii] = dummy[indices_R, :][:, indices_R]

    if ns == 2:
        file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_zscored_2sessions.hdf5')
        fc_all_s1 = file['session_1']['fc'].value
        fc_all_s2 = file['session_2']['fc'].value
        fc_l_4 = np.empty((180, 180, 2, N))
        fc_r_4 = np.empty((180, 180, 2, N))
        for ii in range(N):
            dummy = fc_all_s1[:, :, ii]
            fc_l_4[:, :, 0, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 0, ii] = dummy[indices_R, :][:, indices_R]
            dummy = fc_all_s2[:, :, ii]
            fc_l_4[:, :, 1, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 1, ii] = dummy[indices_R, :][:, indices_R]
    else:
        file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_demean_4_sessions.hdf5')
        fc_all_s1 = file['rest_1']['fc'].value
        fc_all_s2 = file['rest_2']['fc'].value
        fc_all_s3 = file['rest_3']['fc'].value
        fc_all_s4 = file['rest_4']['fc'].value
        fc_l_4 = np.empty((180, 180, 4, N))
        fc_r_4 = np.empty((180, 180, 4, N))
        for ii in range(N):
            dummy = fc_all_s1[:, :, ii]
            fc_l_4[:, :, 0, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 0, ii] = dummy[indices_R, :][:, indices_R]
            dummy = fc_all_s2[:, :, ii]
            fc_l_4[:, :, 1, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 1, ii] = dummy[indices_R, :][:, indices_R]
            dummy = fc_all_s3[:, :, ii]
            fc_l_4[:, :, 2, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 2, ii] = dummy[indices_R, :][:, indices_R]
            dummy = fc_all_s4[:, :, ii]
            fc_l_4[:, :, 3, ii] = dummy[indices_L, :][:, indices_L]
            fc_r_4[:, :, 3, ii] = dummy[indices_R, :][:, indices_R]

    file.close()
    return fc_l, fc_r, fc_l_4, fc_r_4


"""
Collect all subjects
"""
def get_all_subjects():
    from tools.linalg import subdiag

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_zscored_2sessions.hdf5')
    fc_s1 = file['session_1']['fc'].value
    fc_s2 = file['session_2']['fc'].value
    file.close()

    fc_ind_left_s1 = np.zeros((90 * 179, 334))
    fc_ind_right_s1 = np.zeros((90 * 179, 334))
    fc_ind_left_s2 = np.zeros((90 * 179, 334))
    fc_ind_right_s2 = np.zeros((90 * 179, 334))
    fc_ind_whole_s1 = np.zeros((2 * 90 * 179, 334))
    fc_ind_whole_s2 = np.zeros((2 * 90 * 179, 334))

    for ii in range(334):
        dummy_s1 = np.copy(fc_s1[:, :, ii])
        dummy_s2 = np.copy(fc_s2[:, :, ii])
        dummy_s1l = dummy_s1[indices_L, :][:, indices_L]
        dummy_s1r = dummy_s1[indices_R, :][:, indices_R]
        dummy_s2l = dummy_s2[indices_L, :][:, indices_L]
        dummy_s2r = dummy_s2[indices_R, :][:, indices_R]

        fc_ind_left_s1[:,ii] = subdiag(dummy_s1l)
        fc_ind_left_s2[:, ii] = subdiag(dummy_s2l)
        fc_ind_right_s1[:, ii] = subdiag(dummy_s1r)
        fc_ind_right_s2[:, ii] = subdiag(dummy_s2r)

        fc_ind_whole_s1[:, ii] = np.hstack((subdiag(dummy_s1l), subdiag(dummy_s1r)))
        fc_ind_whole_s2[:, ii] = np.hstack((subdiag(dummy_s2l), subdiag(dummy_s2r)))
        print ii

    fout = data_input.save('hcp_334_cortex_zscored_two_sessions.hdf5')
    g1 = fout.create_group('session_1')
    g1.create_dataset('fc', data=fc_ind_whole_s1, compression="lzf")

    g1l = g1.create_group('L')
    g1l.create_dataset('fc', data=fc_ind_left_s1, compression="lzf")

    g1r = g1.create_group('R')
    g1r.create_dataset('fc', data=fc_ind_right_s1, compression="lzf")

    g2 = fout.create_group('session_2')
    g2.create_dataset('fc', data=fc_ind_whole_s2, compression="lzf")

    g2l = g2.create_group('L')
    g2l.create_dataset('fc', data=fc_ind_left_s2, compression="lzf")

    g2r = g2.create_group('R')
    g2r.create_dataset('fc', data=fc_ind_right_s2, compression="lzf")

    fout.close()


"""
Collect all subjects
"""
def get_demean_sessions():
    from tools.linalg import subdiag

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/hcp_334_connectivity_cortex_demean_4_sessions.hdf5')
    #fc_s1 = file['session_1']['fc'].value
    #fc_s2 = file['session_2']['fc'].value


    import pdb; pdb.set_trace()
    file.close()


"""
Collect single session
"""
def get_single_session():
    from tools.linalg import subdiag

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']


    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file = data_input.load('glasser_cole/processed/hcp_334_connectivity_cortex_zscored_single_session.hdf5')
    fc = file['fc'].value
    psd = file['psd'].value
    tau = file['tau'].value
    freqs = file['freqs'].value
    lags = file['lags'].value

    file.close()

    fc_ind_left = np.zeros((90 * 179, 334))
    fc_ind_right = np.zeros((90 * 179, 334))
    fc_ind_whole_lr = np.zeros((2 * 90 * 179, 334))
    fc_ind_whole = np.zeros((180 * 359, 334))

    fc_l_ind = np.empty((180, 180, 334))
    fc_r_ind = np.empty((180, 180, 334))

    for ii in range(334):
        dummy = np.copy(fc[:, :, ii])
        dummy_l = dummy[indices_L,:][:,indices_L]
        dummy_r = dummy[indices_R,:][:,indices_R]

        fc_l_ind[:, :, ii] = np.copy(dummy_l)
        fc_r_ind[:, :, ii] = np.copy(dummy_r)

        fc_ind_left[:,ii] = subdiag(dummy_l)
        fc_ind_right[:, ii] = subdiag(dummy_r)
        fc_ind_whole_lr[:, ii] = np.hstack((subdiag(dummy_l), subdiag(dummy_r)))
        fc_ind_whole[:, ii] = subdiag(dummy)
        print ii

    file_meg = data_input.load('glasser_cole/processed/hcp_meg_csd.hdf5')
    av_psd_meg = np.abs(file_meg['psd'].value)
    freqs_meg = file_meg['freqs'].value
    file_meg.close()


    fout = data_input.save('hcp_334_cortex_zscored_single_session.hdf5')
    fout.create_dataset('fc_whole', data=fc_ind_whole, compression="lzf")
    fout.create_dataset('fc', data=fc_ind_whole_lr, compression="lzf")
    fout.create_dataset('fc_mat', data=fc, compression="lzf")

    fout.create_dataset('psd', data=psd.mean(2), compression="lzf")
    fout.create_dataset('tau', data=tau, compression="lzf")
    fout.create_dataset('freqs', data=freqs, compression="lzf")
    fout.create_dataset('lags', data=lags, compression="lzf")

    fout.create_dataset('av_psd_meg', data=av_psd_meg, compression="lzf")
    fout.create_dataset('freqs_meg', data=freqs_meg, compression="lzf")

    gl = fout.create_group('L')
    gl.create_dataset('fc', data=fc_ind_left, compression="lzf")
    gl.create_dataset('fc_mat', data=fc_l_ind, compression="lzf")

    gr = fout.create_group('R')
    gr.create_dataset('fc', data=fc_ind_right, compression="lzf")
    gr.create_dataset('fc_mat', data=fc_r_ind, compression="lzf")

    fout.close()

def get_autocorrelations():

    from tools.timeseries import TimeSeries
    '''
    Define data structure
    '''
    data = Data(input_dir=data_dir, output_dir=out_dir)
    sc_file = data.load('glasser_cole/raw/DWI_WholeBrain.hdf5') # SC matrices
    fc_file = data.load('glasser_cole/raw/BOLD_MSMAll_hp2000_clean.hdf5') # Time-series

    subjects = utils.hcp_subjects(sc_file)
    N_subjects = len(subjects)
    variables = utils.init_vars(['fc'], (360, 360, N_subjects))
    variables = utils.init_vars(['tau'], (360, N_subjects), variables)
    variables = utils.init_vars(['psd'], (360, 100, N_subjects), variables)

    auto_correlations = np.empty((360, 6, 334))

    time_lags = np.arange(0,6)*0.72
    freq_ds = 22

    ## For subjects
    for ii, ss in enumerate(subjects):
        # FC matrix for session 1 and session 2
        # get FC matrix
        ts = TimeSeries(utils.hcp_get(fc_file, ss, ['rest_1', 'rest_2', 'rest_3', 'rest_4']), fs=0.72, standardize=True, demean=False, gsr=False, index=19, crop=100)
        ts._ts = ts._standardize(ts.ts)
        variables['fc'][:,:,ii] = ts.fc
        autocorr = np.ones(360)
        for ll in np.arange(1,6):
            autocorr = np.vstack((autocorr, ts.auto_corr(ll)))

        auto_correlations[:,:,ii] = np.copy(autocorr.T)

        print 'subject ' + str(ii)

    fname = 'hcp_single_session_autocorrelations.hdf5'
    file_hdf = data.save(fname)
    file_hdf.create_dataset('auto_correlations', data=auto_correlations,compression="lzf")
    file_hdf.create_dataset('lags', data=time_lags,compression="lzf")

    file_hdf.close()


def get_noise():
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    fin = data_input.load('glasser_cole/noise_processed.hdf5')
    subjects = fin['subjects'].value

    av_abs_movement = np.zeros((4, len(subjects)))
    std_abs_movement = np.zeros((4, len(subjects)))
    av_rel_movement = np.zeros((4, len(subjects)))
    std_rel_movement = np.zeros((4, len(subjects)))
    av_hr_var = np.zeros((4, len(subjects)))
    av_resp_var = np.zeros((4, len(subjects)))

    for ii in range(len(subjects)):
        av_abs_movement[:, ii] = np.array([fin[str(subjects[ii])]['AbsoluteMovement_1'].value.mean(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_2'].value.mean(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_3'].value.mean(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_4'].value.mean()])

        std_abs_movement[:, ii] = np.array([fin[str(subjects[ii])]['AbsoluteMovement_1'].value.std(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_2'].value.std(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_3'].value.std(),
                                           fin[str(subjects[ii])]['AbsoluteMovement_4'].value.std()])

        av_rel_movement[:, ii] = np.array([fin[str(subjects[ii])]['RelativeMovement_1'].value.mean(),
                                           fin[str(subjects[ii])]['RelativeMovement_2'].value.mean(),
                                           fin[str(subjects[ii])]['RelativeMovement_3'].value.mean(),
                                           fin[str(subjects[ii])]['RelativeMovement_4'].value.mean()])

        std_rel_movement[:, ii] = np.array([fin[str(subjects[ii])]['RelativeMovement_1'].value.std(),
                                           fin[str(subjects[ii])]['RelativeMovement_2'].value.std(),
                                           fin[str(subjects[ii])]['RelativeMovement_3'].value.std(),
                                           fin[str(subjects[ii])]['RelativeMovement_4'].value.std()])

        av_hr_var[:, ii] = np.array([fin[str(subjects[ii])]['heartrate_var_1'].value.mean(),
                                           fin[str(subjects[ii])]['heartrate_var_2'].value.mean(),
                                           fin[str(subjects[ii])]['heartrate_var_3'].value.mean(),
                                           fin[str(subjects[ii])]['heartrate_var_4'].value.mean()])

        av_resp_var[:, ii] = np.array([fin[str(subjects[ii])]['respiration_var_1'].value.mean(),
                                           fin[str(subjects[ii])]['respiration_var_2'].value.mean(),
                                           fin[str(subjects[ii])]['respiration_var_3'].value.mean(),
                                           fin[str(subjects[ii])]['respiration_var_4'].value.mean()])

    fname = 'hcp_artifacts.hdf5'
    file_hdf = data_input.save(fname)
    file_hdf.create_dataset('av_absolute_movement', data=av_abs_movement,compression="lzf")
    file_hdf.create_dataset('std_absolute_movement', data=std_abs_movement, compression="lzf")
    file_hdf.create_dataset('av_relative_movement', data=av_rel_movement, compression="lzf")
    file_hdf.create_dataset('std_relative_movement', data=std_rel_movement, compression="lzf")
    file_hdf.create_dataset('av_hr_variability', data=av_hr_var, compression="lzf")
    file_hdf.create_dataset('av_resp_variability', data=av_resp_var, compression="lzf")

    file_hdf.close()


def get_iiv():

    from tools.parcels import Parcel
    glasser = Parcel()
    glasser.drop(na=False, key='surface', value='subcortex')

    indices_L = glasser.parcel.index[glasser.parcel.hemi == 'L']
    indices_R = glasser.parcel.index[glasser.parcel.hemi == 'R']

    N = 334
    fc_l, fc_r, fc_l_2s, fc_r_2s = get_empirical(N, 2)
    fc_l, fc_r, fc_l_4s, fc_r_4s = get_empirical(N, 4)

    ''' Inter-subject variability '''
    """
    Single session
    """
    iiv_1_l = (1.0 - indv_variability(fc_l)).mean(1)
    iiv_1_r = (1.0 - indv_variability(fc_r)).mean(1)
    iiv_1 = get_full(iiv_1_l, iiv_1_r)

    """
    2 sessions
    """
    iiv_1_l_2s = (1.0 - indv_variability(fc_l_2s[:, :, 0, :])).mean(1)
    iiv_1_r_2s = (1.0 - indv_variability(fc_r_2s[:, :, 0, :])).mean(1)
    iiv_1_2s = get_full(iiv_1_l_2s, iiv_1_r_2s)

    iiv_2_l_2s = (1.0 - indv_variability(fc_l_2s[:, :, 1, :])).mean(1)
    iiv_2_r_2s = (1.0 - indv_variability(fc_r_2s[:, :, 1, :])).mean(1)
    iiv_2_2s = get_full(iiv_2_l_2s, iiv_2_r_2s)

    intra_indv_var_2s_rank = np.empty((360, N))
    for ii in range(N):
        intra_indv_var_2s_rank[indices_L, ii] = 1. - indv_variability(fc_l_2s[:, :, :, ii])
        intra_indv_var_2s_rank[indices_R, ii] = 1. - indv_variability(fc_r_2s[:, :, :, ii])

    """
    4 sessions
    """
    iiv_1_l_4s = (1.0 - indv_variability(fc_l_4s[:, :, 0, :])).mean(1)
    iiv_1_r_4s = (1.0 - indv_variability(fc_r_4s[:, :, 0, :])).mean(1)
    iiv_1_4s = get_full(iiv_1_l_4s, iiv_1_r_4s)

    iiv_2_l_4s = (1.0 - indv_variability(fc_l_4s[:, :, 1, :])).mean(1)
    iiv_2_r_4s = (1.0 - indv_variability(fc_r_4s[:, :, 1, :])).mean(1)
    iiv_2_4s = get_full(iiv_2_l_4s, iiv_2_r_4s)

    iiv_3_l_4s = (1.0 - indv_variability(fc_l_4s[:, :, 2, :])).mean(1)
    iiv_3_r_4s = (1.0 - indv_variability(fc_r_4s[:, :, 2, :])).mean(1)
    iiv_3_4s = get_full(iiv_3_l_4s, iiv_3_r_4s)

    iiv_4_l_4s = (1.0 - indv_variability(fc_l_4s[:, :, 3, :])).mean(1)
    iiv_4_r_4s = (1.0 - indv_variability(fc_r_4s[:, :, 3, :])).mean(1)
    iiv_4_4s = get_full(iiv_4_l_4s, iiv_4_r_4s)

    intra_indv_var_4s_rank = np.empty((360, N))
    for ii in range(N):
        intra_indv_var_4s_rank[indices_L, ii] = 1. - indv_variability(fc_l_4s[:, :, :, ii]).mean(1)
        intra_indv_var_4s_rank[indices_R, ii] = 1. - indv_variability(fc_r_4s[:, :, :, ii]).mean(1)

    # import statsmodels.api as sm
    # model = sm.OLS(inter_indv_var, sm.add_constant(intra_indv_var.mean(1), prepend=False), missing='drop')
    # results = model.fit()

    """
    Write results
    """
    data_input = Data(input_dir=data_dir, output_dir=out_dir)
    file_out = data_input.save('individual_variability_empirical.hdf5')
    file_out.create_dataset('inter_indv_var', data=iiv_1, compression='lzf')

    g2s = file_out.create_group('2_sessions')
    g2s.create_dataset('intra_indv_var', data=intra_indv_var_2s_rank, compression='lzf')
    g2s.create_dataset('inter_indv_var_s1', data=iiv_1_2s, compression='lzf')
    g2s.create_dataset('inter_indv_var_s2', data=iiv_2_2s, compression='lzf')

    g4s = file_out.create_group('4_sessions')
    g4s.create_dataset('intra_indv_var', data=intra_indv_var_4s_rank, compression='lzf')
    g4s.create_dataset('inter_indv_var_s1', data=iiv_1_4s, compression='lzf')
    g4s.create_dataset('inter_indv_var_s2', data=iiv_2_4s, compression='lzf')
    g4s.create_dataset('inter_indv_var_s3', data=iiv_3_4s, compression='lzf')
    g4s.create_dataset('inter_indv_var_s4', data=iiv_4_4s, compression='lzf')

    file_out.close()

data_dir = '/Users/md2242/Projects/hcp_tools/data/'
out_dir = '/Users/md2242/Projects/hcp_tools/output/forHeterogeneity'


get_demean_sessions()
#get_single_session()
#get_noise()
#get_all_subjects()
#get_single_session()
#get_autocorrelations()