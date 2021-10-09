import os
import json
import h5py
import gzip
import io
import copy
import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
from utils.image_utils import get_spatial_feat
from torch.utils.data import Dataset
from transformers import BertTokenizer
import collections


class OracleDataset(Dataset):
    def __init__(self, data_dir, data_file, split, visual_feat_file, visual_feat_mapping_file,
                 visual_feat_crop_file, visual_feat_crop_mapping_file, max_src_length, hdf5_visual_feat,
                 hdf5_crop_feat, history = False, new_oracle_data=False, successful_only=True, min_occ=3, load_crops=False, bert_tok=False, only_location=False):

        self.data_dir = data_dir
        self.split = split
        self.history = history
        # where to save/load preprocessed data
        if self.history:
            self.data_file_name = 'oracle_' + split + '_history_data.json'
        else:
            if bert_tok:
                self.data_file_name = 'oracle_' + split + '_bert_tok_data.json'
            elif only_location:
                self.data_file_name = 'oracle_' + split + '_location_data.json'
            else:
                self.data_file_name = 'oracle_' + split + '_data.json'

        print("data file name", self.data_file_name)

        self.vocab_file_name = 'vocab.json'
        self.min_occ = min_occ
        # original guesswhat data
        self.data_file = data_file
        self.visual_feat_file = os.path.join(data_dir, visual_feat_file)
        self.visual_feat_crop_file = os.path.join(data_dir, visual_feat_crop_file)
        self.max_src_length = max_src_length
        self.hdf5_visual_feat = hdf5_visual_feat
        self.hdf5_crop_feat = hdf5_crop_feat
        self.successful_only = successful_only
        self.load_crops = load_crops

        if self.history:
            self.max_diag_len = self.max_src_length*2+1
        else:
            self.max_diag_len = None

        self.vf = np.asarray(h5py.File(self.visual_feat_file, 'r')[self.hdf5_visual_feat])
        if self.load_crops:
            # self.cf = np.asarray(h5py.File(self.visual_feat_crop_file, 'r')[self.hdf5_crop_feat])
            self.cf = np.asarray(h5py.File('./data/target_objects_features_fast.h5', 'r')['objects_features'])

            # with open(os.path.join(data_dir, visual_feat_crop_mapping_file), 'r') as file_c:
            #     self.visual_feat_crop_mapping_file = json.load(file_c)
            # self.visual_feat_crop_mapping_file = self.visual_feat_crop_mapping_file['crops_features2id']
            with open('./data/target_objects_features_index_fast.json', 'r') as file_c:
                self.visual_feat_crop_mapping_file = json.load(file_c)

        with open(os.path.join(data_dir, visual_feat_mapping_file), 'r') as file_v:
            self.visual_feat_mapping_file = json.load(file_v)

        self.visual_feat_mapping_file = self.visual_feat_mapping_file[split+'2id']

        # load or create new vocab
        if bert_tok:
            with open('./data/vocab_bert_tok.json', 'r') as file:
                self.word2i = json.load(file)['word2i']
        else:
            with open(os.path.join(data_dir, self.vocab_file_name), 'r') as file:
                self.word2i = json.load(file)['word2i']

        # create new oracle_data file or load from disk
        if not os.path.isfile(os.path.join(self.data_dir, self.data_file_name)) or new_oracle_data:
            if bert_tok:
                self.oracle_data = self.new_oracle_data_bert_tok()
            if only_location:
                self.oracle_data = self.new_oracle_data_location()
            else:
                self.oracle_data = self.new_oracle_data()
        else:
            print("reading", os.path.join(self.data_dir, self.data_file_name))
            with open(os.path.join(self.data_dir, self.data_file_name), 'r') as file:
                self.oracle_data = json.load(file)

    def __len__(self):
        return len(self.oracle_data)

    def __getitem__(self, idx):

        if not type(idx) == str:
            idx = str(idx)

        # load image features
        visual_feat_id = self.visual_feat_mapping_file[self.oracle_data[idx]['image_file']]
        visual_feat = self.vf[visual_feat_id]
        if self.load_crops:
            # crop_feat_id = self.visual_feat_crop_mapping_file[self.oracle_data[idx]['game_id']+'.jpg']
            crop_feat_id = self.visual_feat_crop_mapping_file[self.oracle_data[idx]['game_id']]
            crop_feat = self.cf[crop_feat_id]
        else:
            crop_feat = 0

        return {'question': np.asarray(self.oracle_data[idx]['question']),
                'answer': self.oracle_data[idx]['answer'],
                'crop_features': crop_feat,
                'img_features': visual_feat,
                'spatial': np.asarray(self.oracle_data[idx]['spatial'], dtype=np.float32),
                'obj_cat': self.oracle_data[idx]['obj_cat'],
                'length': self.oracle_data[idx]['length'],
                'game_id': self.oracle_data[idx]['game_id']}

    def new_oracle_data(self):

        print("Creating New " + self.data_file_name + " File.")

        path = os.path.join(self.data_dir, self.data_file)
        tknzr = TweetTokenizer(preserve_case=False)
        oracle_data = dict()
        _id = 0

        ans2tok = {'Yes': 1,
                   'No': 0,
                   'N/A': 2}

        with gzip.open(path) as file:
            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only:
                    if not game['status'] == 'success':
                        continue

                if self.history:
                    prev_ques = list()
                    prev_answer = list()
                    prev_length = 0
                for i, qa in enumerate(game['qas']):
                    q_tokens = tknzr.tokenize(qa['question'])
                    q_token_ids = [self.word2i[w] if w in self.word2i else self.word2i['<unk>']for w in q_tokens][:self.max_src_length]
                    a_token = ans2tok[qa['answer']]

                    length = len(q_token_ids)

                    if self.history:
                        question = prev_ques+prev_answer+q_token_ids
                        question_length = prev_length+length
                    else:
                        question = q_token_ids
                        question_length = length

                    if self.history:
                        question.extend([self.word2i['<padding>']] * (self.max_diag_len - len(question)))
                    else:
                        question.extend([self.word2i['<padding>']] * (self.max_src_length - len(question)))

                    for i, o in enumerate(game['objects']):
                        if o['id'] == game['object_id']:
                            # target object information
                            spatial = get_spatial_feat(bbox=o['bbox'], im_width=game['image']['width'], im_height=game['image']['height'])
                            object_category = o['category_id']
                            break

                    oracle_data[_id]                = dict()
                    oracle_data[_id]['question']    = question
                    oracle_data[_id]['length']      = question_length
                    oracle_data[_id]['answer']      = a_token
                    oracle_data[_id]['image_file']  = game['image']['file_name']
                    oracle_data[_id]['spatial']     = spatial
                    oracle_data[_id]['game_id']     = str(game['id'])
                    oracle_data[_id]['obj_cat']     = object_category

                    prev_ques = copy.deepcopy(q_token_ids)
                    prev_answer = [copy.deepcopy(a_token)]
                    prev_length = length+1

                    _id += 1

        oracle_data_path = os.path.join(self.data_dir, self.data_file_name)
        with io.open(oracle_data_path, 'wb') as f_out:
            data = json.dumps(oracle_data, ensure_ascii=False)
            f_out.write(data.encode('utf8', 'replace'))

        print('done')

        with open(oracle_data_path, 'r') as file:
            oracle_data = json.load(file)

        return oracle_data

    def new_oracle_data_bert_tok(self):

        print("Creating New " + self.data_file_name + " File.")

        path = os.path.join(self.data_dir, self.data_file)
        tknzr = BertTokenizer.from_pretrained("bert-base-uncased")
        oracle_data = dict()
        _id = 0

        ans2tok = {'Yes': 1,
                   'No': 0,
                   'N/A': 2}


        with gzip.open(path) as file:
            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only:
                    if not game['status'] == 'success':
                        continue

                if self.history:
                    prev_ques = list()
                    prev_answer = list()
                    prev_length = 0
                for i, qa in enumerate(game['qas']):
                    q_tokens = tknzr.tokenize(qa['question'])
                    q_token_ids = [self.word2i[w] if w in self.word2i else self.word2i['[UNK]'] for w in q_tokens][:self.max_src_length]
                    a_token = ans2tok[qa['answer']]

                    length = len(q_token_ids)

                    if self.history:
                        question = prev_ques+prev_answer+q_token_ids
                        question_length = prev_length+length
                    else:
                        question = q_token_ids
                        question_length = length

                    if self.history:
                        question.extend([self.word2i['<padding>']] * (self.max_diag_len - len(question)))
                    else:
                        question.extend([self.word2i['<padding>']] * (self.max_src_length - len(question)))

                    for i, o in enumerate(game['objects']):
                        if o['id'] == game['object_id']:
                            # target object information
                            spatial = get_spatial_feat(bbox=o['bbox'], im_width=game['image']['width'], im_height=game['image']['height'])
                            object_category = o['category_id']
                            break

                    oracle_data[_id]                = dict()
                    oracle_data[_id]['question']    = question
                    oracle_data[_id]['length']      = question_length
                    oracle_data[_id]['answer']      = a_token
                    oracle_data[_id]['image_file']  = game['image']['file_name']
                    oracle_data[_id]['spatial']     = spatial
                    oracle_data[_id]['game_id']     = str(game['id'])
                    oracle_data[_id]['obj_cat']     = object_category

                    prev_ques = copy.deepcopy(q_token_ids)
                    prev_answer = [copy.deepcopy(a_token)]
                    prev_length = length+1

                    _id += 1

        oracle_data_path = os.path.join(self.data_dir, self.data_file_name)
        with io.open(oracle_data_path, 'wb') as f_out:
            data = json.dumps(oracle_data, ensure_ascii=False)
            f_out.write(data.encode('utf8', 'replace'))

        print('done')

        with open(oracle_data_path, 'r') as file:
            oracle_data = json.load(file)

        return oracle_data

    def new_oracle_data_location(self):

        print("Creating New " + self.data_file_name + " File.")

        with open("data/word_annotation") as f:
            lines = f.readlines()
            word2cat = {}
            for line in lines:
                word1, word2 = line.split("\t")
                word1 = word1.strip().lower()
                word2 = word2.strip().lower()
                if word2 == "color":
                    word2cat[word1] = "color"
                elif word2 == "shape":
                    word2cat[word1] = "shape"
                elif word2 == "size":
                    word2cat[word1] = "size"
                elif word2 == "texture":
                    word2cat[word1] = "texture"
                elif word2 == "action":
                    word2cat[word1] = "action"
                elif word2 == "spatial":
                    word2cat[word1] = "spatial"
                elif word2 == "number":
                    word2cat[word1] = "number"
                elif word2 == "object":
                    word2cat[word1] = "object"
                elif word2 == "super-category":
                    word2cat[word1] = "super-category"

        path = os.path.join(self.data_dir, self.data_file)
        tknzr = TweetTokenizer(preserve_case=False)
        oracle_data = dict()
        _id = 0
        count = collections.defaultdict(int)

        ans2tok = {'Yes': 1,
                   'No': 0,
                   'N/A': 2}

        reverse_ans = {1: 0, 0: 1, 2: 2}
        reverse_token = {64: 26, 26: 64, 37: 172, 172: 37,16:137,137:16}
        list_rev_tok = [64,26,37,172,16,137]
        original_data = 0
        generated_data = 0

        with gzip.open(path) as file:
            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only:
                    if not game['status'] == 'success':
                        continue

                if self.history:
                    prev_ques = list()
                    prev_answer = list()
                    prev_length = 0
                for i, qa in enumerate(game['qas']):
                    valid = False
                    q_tokens = tknzr.tokenize(qa['question'])
                    for tok in q_tokens:
                        if tok in word2cat and word2cat[tok] == 'spatial':
                            valid = True
                            count[tok] += 1
                    if not valid:
                        continue
                    q_token_ids = [self.word2i[w] if w in self.word2i else self.word2i['<unk>']for w in q_tokens][:self.max_src_length]
                    a_token = ans2tok[qa['answer']]

                    length = len(q_token_ids)

                    if self.history:
                        question = prev_ques+prev_answer+q_token_ids
                        question_length = prev_length+length
                    else:
                        question = q_token_ids
                        question_length = length

                    if self.history:
                        question.extend([self.word2i['<padding>']] * (self.max_diag_len - len(question)))
                    else:
                        question.extend([self.word2i['<padding>']] * (self.max_src_length - len(question)))

                    for i, o in enumerate(game['objects']):
                        if o['id'] == game['object_id']:
                            # target object information
                            spatial = get_spatial_feat(bbox=o['bbox'], im_width=game['image']['width'], im_height=game['image']['height'])
                            object_category = o['category_id']
                            break

                    oracle_data[_id]                = dict()
                    oracle_data[_id]['question']    = question
                    oracle_data[_id]['length']      = question_length
                    oracle_data[_id]['answer']      = a_token
                    oracle_data[_id]['image_file']  = game['image']['file_name']
                    oracle_data[_id]['spatial']     = spatial
                    oracle_data[_id]['game_id']     = str(game['id'])
                    oracle_data[_id]['obj_cat']     = object_category

                    prev_ques = copy.deepcopy(q_token_ids)
                    prev_answer = [copy.deepcopy(a_token)]
                    prev_length = length+1

                    _id += 1
                    original_data+= 1

                    # if self.split == 'train' and any(i in list_rev_tok for i in question) and a_token==1:
                    #     new_question = []
                    #     for tok in question:
                    #         if tok in reverse_token:
                    #             new_question.append(reverse_token[tok])
                    #         else:
                    #             new_question.append(tok)
                    #     new_a_token = reverse_ans[a_token]
                    #
                    #     oracle_data[_id] = dict()
                    #     oracle_data[_id]['question'] = new_question
                    #     oracle_data[_id]['length'] = question_length
                    #     oracle_data[_id]['answer'] = new_a_token
                    #     oracle_data[_id]['image_file'] = game['image']['file_name']
                    #     oracle_data[_id]['spatial'] = spatial
                    #     oracle_data[_id]['game_id'] = str(game['id'])
                    #     oracle_data[_id]['obj_cat'] = object_category
                    #     _id += 1
                    #     generated_data += 1


        oracle_data_path = os.path.join(self.data_dir, self.data_file_name)
        with io.open(oracle_data_path, 'wb') as f_out:
            data = json.dumps(oracle_data, ensure_ascii=False)
            f_out.write(data.encode('utf8', 'replace'))

        print("original data {}, generated data {}".format(original_data, generated_data))
        print('done')

        with open(oracle_data_path, 'r') as file:
            oracle_data = json.load(file)

        return oracle_data
