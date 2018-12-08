from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset


class KITTI(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,mode,
                 input_res,
                 db_root_dir,
                 central_frame_list,
                 path_to_depth_npy,
                 path_to_valid_files,
                 seq_length):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.mode = mode
        self.input_res = input_res
        self.db_root_dir = db_root_dir
        self.file_index_dict = {}
        self.populate_helper_ds(path_to_depth_npy,path_to_valid_files)
        self.seq_length = seq_length
        with open(central_frame_list, 'r') as f:
            self.central_frame_paths = f.readlines()

        '''
        #Siddhanj: When there is a sequence name given, we want to load just that one sequence, else, we want to load all sequences
        #Maybe for online training, we will write a separate data loader? For now focus is on just loading the entire sequence

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir,'ImageSets',year, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:
            seqs = [seq_name]
            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            # if self.train:
            #     img_list = [img_list[0]]
            #     labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.seqs = seqs
        print('Done initializing ' + fname + ' Dataset')
        '''
    def populate_helper_ds(self,path_to_depth_npy,path_to_valid_files):
        self.depth_npy = np.load(path_to_depth_npy)
        with open(path_to_valid_files, 'r') as f:
            self.files_paths = f.readlines()

        for idx in len(self.files_paths):
            self.file_index_dict[self.files_paths[idx]] = idx

    def __len__(self):
        return len(self.central_frame_paths)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        sample = {'image': img, 'gt': gt}
        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the images-ground-truth pair
        """

        central_frame_file_path = self.central_frame_paths[idx]
        date,drive,_,_,frame_file_name = central_frame_file_path.split("/")
        frame_idx = int(frame_file_name.split(".")[0])

        half_offset = int((self.seq_length - 1) / 2)
        min_src_idx = frame_idx - half_offset
        max_src_idx = frame_idx + half_offset +1

        total_num_of_frames = self.seq_length
        img_size = self.get_img_size()
        imgs = np.zeros([total_num_of_frames, 3, img_size[0], img_size[1]], dtype=np.float32)
        gts = np.zeros([total_num_of_frames, 1, img_size[0], img_size[1]], dtype=np.float32)

        last_read_frame_idx = frame_idx
        global_ctr = 0
        for leftCtr in range(frame_idx,min_src_idx-1,-1):
            curr_frame_idx = leftCtr
            if curr_frame_idx < 0:
                curr_frame_idx = last_read_frame_idx

            frame_file_name = '%.10d' % (np.int(curr_frame_idx)) + '.jpg'
            img_file_path = os.path.join(self.db_root_dir,date,drive,'image_02/data',frame_file_name)
            img_file_idx = self.file_index_dict[img_file_path]

            c_img = cv2.imread(img_file_path)
            c_img = np.transpose(c_img, (2, 0, 1))
            c_label = self.depth_npy[img_file_idx,:,:]

            if self.inputRes is not None:
                c_img = imresize(c_img, self.inputRes)
                c_label = imresize(c_label, self.inputRes, interp='nearest')

            imgs[global_ctr, :, :, :] = c_img
            gts[global_ctr, :, :, :] = c_label
            last_read_frame_idx = curr_frame_idx
            global_ctr = global_ctr + 1

        last_read_frame_idx = frame_idx
        for rightCtr in range(frame_idx+1, max_src_idx+1, 1):
            curr_frame_idx = rightCtr
            frame_file_name = '%.10d' % (np.int(curr_frame_idx)) + '.jpg'
            img_file_path = os.path.join(self.db_root_dir, date, drive, 'image_02/data', frame_file_name)

            if  os.path.exists(img_file_path) == False:
                curr_frame_idx = last_read_frame_idx

            frame_file_name = '%.10d' % (np.int(curr_frame_idx)) + '.jpg'
            img_file_path = os.path.join(self.db_root_dir, date, drive, 'image_02/data', frame_file_name)
            img_file_idx = self.file_index_dict[img_file_path]

            c_img = cv2.imread(img_file_path)
            c_img = np.transpose(c_img, (2, 0, 1))
            c_label = self.depth_npy[img_file_idx, :, :]

            if self.inputRes is not None:
                c_img = imresize(c_img, self.inputRes)
                c_label = imresize(c_label, self.inputRes, interp='nearest')

            imgs[global_ctr, :, :, :] = c_img
            gts[global_ctr, :, :, :] = c_label
            last_read_frame_idx = curr_frame_idx

            global_ctr = global_ctr + 1


        imgs = np.array(imgs, dtype=np.float32)
        gts = np.array(gts, dtype=np.float32)

        #SiddhantGeo: Normalise/log normalise as the need be
        #gts = gts / np.max([gts.max(), 1e-8])

        return imgs, gts

    def get_img_size(self):
        return [self.input_res[0],self.input_res[1]]


if __name__ == '__main__':

    import torch

    from matplotlib import pyplot as plt


    #siddhanj: scale messes is it up for somereason. Investigate into this later
    #transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])
    #transforms = transforms.Compose([tr.VideoLucidDream() , tr.ToTensor()])
    #transforms = transforms.Compose([tr.ToTensor()])

    dataset = KITTI(mode='train',
                 input_res=[512,348],
                 db_root_dir='/home/siddhanj/16822/monodepth-data',
                 central_frame_list='train_file_list.txt',
                 path_to_depth_npy='/home/siddhanj/16822/code/monodepth/models/disparities_pp.npy',
                 path_to_valid_files='allfiles.txt',
                 seq_length=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        img = data['image'][0,0,:,:,:]
        label = data['gt'][0,0,:,:,:]



