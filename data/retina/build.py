import numpy as np
import os
import re
import sys
import zipfile
import urllib.request
DATA_URL = 'http://cvlabwww.epfl.ch/~sznitman/retinal_dataset.zip'
DATA_DIR = 'retinal_dataset/retinal_dataset/'
seqs = ['seq1.txt', 'seq2.txt', 'seq3.txt']

def maybe_download_and_extract():
    dest_directory = "./retinal_dataset"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                     float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


def build_set(name):

    for si in range(0, len(seqs)):
        print("loading seq: " + seqs[si])
        seq_data = np.genfromtxt(DATA_DIR + seqs[si], dtype=np.int32, delimiter=" ")
        filenames = [DATA_DIR + s.zfill(6) + ".png" for s in map(str, seq_data[:, 0], )]
        poses = seq_data[:, 1:]
        objects = np.ones([len(filenames), 1])
        sequence = np.ones([len(filenames), 1]) * si
        for i in range(len(filenames)):
            if np.any(poses[i, :] == -1):
                objects[i, 0] = 0


        filenames = np.expand_dims(filenames, axis=1)
        out = np.hstack((np.hstack((np.hstack((filenames, sequence)), objects)), poses))
        for i in range(len(filenames)-1, -1 , -1):
            filename = out[i,0]

            if(os.path.isfile(filename)==False ):
                out = np.delete(out, i, axis=0)
            elif(float(out[i,2]) < 1):
                out = np.delete(out, i, axis=0)

        try:
            out_tot = np.vstack((out_tot,out))
        except NameError:
            out_tot = out

    np.savetxt(name + ".csv", out_tot, delimiter=",", fmt="%s")





def main(argv=None):  # pylint: disable=unused-argument
    maybe_download_and_extract()
    build_set("retinal_dataset")

if __name__ == "__main__":
    main()
