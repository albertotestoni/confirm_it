import os
import json
import numpy as np
import h5py
from PIL import Image
# from utils.datasets.GamePlay.prepro import create_data_file
from torch.utils.data import Dataset
from torchvision import transforms

class GamePlayDataset_NewOracle(Dataset):
    """docstring for GameplayN2NResNet."""
    def __init__(self, split="train", with_objects_feat=False, **kwargs):
        super(GamePlayDataset_NewOracle, self).__init__()
        self.data_args = kwargs
        self.with_objects_feat = with_objects_feat
        with open("/mnt/povobackup/clic/alberto.testoni/bboxes.json") as in_file:
            self.bboxes = json.load(in_file)

        if with_objects_feat:
            if split != "test":
                objects_feat_file = os.path.join(self.data_args['data_paths']['ResNet']['objects_features'] )
                objects_feat_mapping_file = os.path.join(self.data_args['data_dir'], self.data_args['data_paths']['ResNet']['objects_features_index'] )
                self.objects_vf = h5py.File(objects_feat_file, 'r')['objects_features']

                with open(objects_feat_mapping_file, 'r') as file_v:
                    self.objects_feat_mapping = json.load(file_v)
            else:
                objects_feat_file = os.path.join(self.data_args['data_paths']['ResNet']['objects_features_test'])
                objects_feat_mapping_file = os.path.join(self.data_args['data_dir'],
                                                         self.data_args['data_paths']['ResNet'][
                                                             'objects_features_index_test'])
                self.objects_vf = h5py.File(objects_feat_file, 'r')['objects_features']

                with open(objects_feat_mapping_file, 'r') as file_v:
                    self.objects_feat_mapping = json.load(file_v)

        self.cf = h5py.File(self.data_args['data_dir'] + 'target_objects_features_all.h5', 'r')['objects_features']
        with open(self.data_args['data_dir'] + '/target_objects_features_index_all.json', 'r') as file_c:
            self.visual_feat_crop_mapping_file = json.load(file_c)

        visual_feat_file = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        tmp_key = split + "_process_file"
        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_'+split+'_successful_gameplay_data.json'
            else:
                data_file_name = 'n2n_'+split+'_all_gameplay_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(data_dir=self.data_args['data_dir'], data_file=self.data_args['data_paths'][split], data_args=self.data_args, vocab_file_name=self.data_args['data_paths']['vocab_file'], split=split)

        with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
            self.game_data = json.load(f)

        for k in self.game_data:
            self.game_data[k]["FasterRCNN"] = self.data_args["imgid2fasterRCNNfeatures"][self.game_data[k]["image_file"].split(".")[0]]

    def __len__(self) :
        return len(self.game_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        # Load image features
        image_file = self.game_data[idx]['image_file']
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['history'] = np.asarray(self.game_data[idx]['history'])
        _data['history_len'] = self.game_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.game_data[idx]['src_q'])
        _data['objects'] = np.asarray(self.game_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.game_data[idx]['objects'], np.zeros(len(self.game_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.game_data[idx]['spatials'])
        _data['target_obj'] = self.game_data[idx]['target_obj']
        _data['target_cat'] = self.game_data[idx]['target_cat']
        _data['target_spatials'] = np.asarray(self.game_data[idx]['target_spatials'], dtype=np.float32)
        _data['image'] = ImgFeat
        _data['image_file'] = image_file
        _data['game_id'] = self.game_data[idx]['game_id']
        _data['image_url'] = self.game_data[idx]['image_url']
        _data["unnormalized_target_bbox"] = np.asarray(self.bboxes[self.game_data[idx]['game_id']], dtype=np.float32)

        crop_feat_id = self.visual_feat_crop_mapping_file[self.game_data[idx]['game_id']]
        crop_feat = self.cf[crop_feat_id]
        _data['crop_features'] = crop_feat

        _data['FasterRCNN'] = dict()

        _data['FasterRCNN']['features'] = self.game_data[idx]['FasterRCNN']['features']
        _data['FasterRCNN']['unnormalized_boxes'] = self.game_data[idx]['FasterRCNN']['boxes']

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = self.game_data[idx]['FasterRCNN']['img_h'], self.game_data[idx]['FasterRCNN']['img_w']
        boxes = self.game_data[idx]['FasterRCNN']['boxes'].copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        _data['FasterRCNN']['boxes'] = boxes

        target_bbox_copy = _data["unnormalized_target_bbox"].copy()
        target_bbox_copy[[0, 2]] /= img_w
        target_bbox_copy[[1, 3]] /= img_h
        _data['target_bbox'] = target_bbox_copy

        # Load object features
        if self.with_objects_feat:
            objects_feat_id = self.objects_feat_mapping[self.game_data[idx]['game_id']]
            objects_feat = self.objects_vf[objects_feat_id]
            _data["objects_feat"] = objects_feat

        return _data
