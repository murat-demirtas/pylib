import os

class WorkbenchWrapper():
    def __init__(self, root_directory, subject_directory_string):
        self.root = root_directory
        self.subject_dir = self.root + subject_directory_string

    def parcellate_dtseries(self, cifti, subject_id, input_file_string, parcel_file, output_file_string):
        if cifti == 'dtseries':
            file_to_parcellate = (self.subject_dir + input_file_string).format(subject=subject_id) + '.dtseries.nii'
            cmd = "wb_command -cifti-parcellate " + file_to_parcellate + ' ' + parcel_file + ' COLUMN ' + output_file_string.format(subject=subject_id) + '.ptseries.nii'
            print cmd
        elif cifti == 'dscalar':
            file_to_parcellate = (self.subject_dir + input_file_string).format(subject=subject_id) + '.dscalar.nii'
            cmd = "wb_command -cifti-parcellate " + file_to_parcellate + ' ' + parcel_file + ' COLUMN ' + output_file_string.format(subject=subject_id) + '.pscalar.nii'
            print cmd
        elif cifti == 'dconn':
            file_to_parcellate = (self.subject_dir + input_file_string).format(subject=subject_id) + '.dconn.nii'
            temp_file = output_file_string.format(subject=subject_id) + '.pdconn.nii'
            cmd = "wb_command -cifti-parcellate " + file_to_parcellate + ' ' + parcel_file + ' COLUMN ' + temp_file

            cmd2 = "wb_command -cifti-parcellate " + temp_file + ' ' + parcel_file + ' ROW ' + output_file_string.format(subject=subject_id) + '.pconn.nii'

            cmd3 = "rm " + temp_file

            print cmd
            print cmd2
            print cmd3

            #os.system('ls')


    def merge(self, input_list):
        print ""


    def average(self, input_list):
        print ""


    def correlate(self, input_file_string, output_file_string):
        print ""


    def variance(self):
        print ""


    def zscore(self, input_file_string):
        print ""


