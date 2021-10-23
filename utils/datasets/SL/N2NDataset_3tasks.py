import json

import h5py
import numpy as np
import os
from torch.utils.data import Dataset


class N2NDataset(Dataset):
    def __init__(self, split='train', **kwargs):
        self.data_args = kwargs
        self.split = split
        visual_feat_file = os.path.join(self.data_args['data_dir'], self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'], self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        tmp_key = split + "_process_file"

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_'+split+'_successful_data_3tasks.json'
            else:
                data_file_name = 'n2n_'+split+'_all_data_3tasks.json'

        if split == 'test':
            data_file_name = 'n2n_' + split + 'data_3tasks_ALL.json'

        print("reading", data_file_name)
        with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
            self.n2n_data = json.load(f)

    def __len__(self):
        return len(self.n2n_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        # Load image features
        image_file = self.n2n_data[idx]['image_file']
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['image'] = ImgFeat
        _data['history'] = np.asarray(self.n2n_data[idx]['history'])
        _data['history_len'] = self.n2n_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.n2n_data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.n2n_data[idx]['target_q'])
        _data['tgt_len'] = self.n2n_data[idx]['tgt_len']
        _data['decider_tgt'] = int(self.n2n_data[idx]['decider_tgt'])
        _data['objects'] = np.asarray(self.n2n_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.n2n_data[idx]['objects'], np.zeros(len(self.n2n_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.n2n_data[idx]['spatials'])
        _data['target_obj'] = self.n2n_data[idx]['target_obj']
        _data['target_cat'] = self.n2n_data[idx]['target_cat']
        _data['game_id'] = self.n2n_data[idx]['game_id']
        _data['bboxes'] = np.asarray(self.n2n_data[idx]['bboxes'])
        _data['image_url'] = self.n2n_data[idx]['image_url']
        _data['oracle_question'] = np.asarray(self.n2n_data[idx]['oracle_question'])
        _data['oracle_length'] = self.n2n_data[idx]['oracle_length']
        _data['oracle_prev_hist'] = np.asarray(self.n2n_data[idx]['oracle_prev_hist'])
        _data['oracle_prev_hist_len'] = self.n2n_data[idx]['oracle_prev_hist_len']

        _data['oracle_answer'] = self.n2n_data[idx]['oracle_answer']
        _data['oracle_spatial'] = np.asarray(self.n2n_data[idx]['oracle_spatial'], dtype="float32")
        _data['oracle_obj_cat'] = self.n2n_data[idx]['oracle_obj_cat']

        if self.split == 'test':
            _data['q_idx'] = self.n2n_data[idx]['q_idx']

        return _data
