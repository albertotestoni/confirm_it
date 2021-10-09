import json
import torch
import collections
import os
import gzip
import io
import h5py
import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from utils.image_utils import get_spatial_feat
import copy

def create_data_file(data_dir, data_file, data_args, vocab_file_name, split='train'):
    """Creates the training/test/val data given dataset file in *.jsonl.gz format.

    Parameters
    ----------
    data_dir : str
        Directory to read the data and dump the training data created
    data_file : str
        Name of the *.jsonl.gz data file
    data_args : dict
        'successful_only' : bool. Checks what type of games to be included.
        'max_no_objects' : int. Number required for padding of objects in target list for Guesser.
        'max_q_length' : int. Max number of words that QGen can use to ask next question
        'max_src_length' : int. Max number of words that can be present in the dialogue history
        'max_no_qs' : int. Max number of questions that a gamme can have to be included in the data
        'data_paths' : str?. Added by ravi for different file name than default. More details to be added by ravi.
    vocab_file_name : str
        vocabulary file name. This file should have 'word2i' and 'i2word'
    split : str
        Split of the data file

    """


    path = os.path.join(data_dir, data_file)
    successful_only = data_args['successful_only']

    tmp_key = split + "_process_file"

    if tmp_key in data_args['data_paths']:
        data_file_name = data_args['data_paths'][tmp_key]
    else:
        if successful_only:
            data_file_name = 'n2n_'+split+'_successful_data_3tasks.json'
            # data_file_name = 'n2n_' + split + '_successful_data_balanced.json'
        else:
            data_file_name = 'n2n_'+split+'_all_data_3tasks.json'

    print("Creating New " + data_file_name + " File.")

    category_pad_token = 0 #TODO Add this to config.json
    decidermask_pad_token = -1 #TODO Add this to config.json
    max_no_objects = data_args['max_no_objects']
    max_q_length = data_args['max_q_length']
    max_src_length = data_args['max_src_length']
    max_no_qs = data_args['max_no_qs']
    no_spatial_feat = 8 #TODO Add this to config.json

    tknzr = TweetTokenizer(preserve_case=False)
    n2n_data = dict()
    _id = 0

    # load or create new vocab
    with open(os.path.join(data_dir, vocab_file_name), 'r') as file:
        word2i = json.load(file)['word2i']

    with open(os.path.join(data_dir, vocab_file_name), 'r') as file:
        i2word = json.load(file)['i2word']

    ans2tok = {'Yes': word2i['<yes>'],
    		   'No': word2i['<no>'],
    		   'N/A': word2i['<n/a>']}

    ans2tok_oracle = {'Yes': 1,
               'No': 0,
               'N/A': 2}

    start = '<start>'
    # stop = '<stop>'
    ans_yes = 0
    ans_no = 0
    ans_na = 0
    freq_dial_len = collections.defaultdict(int)

    def ids2dial(list_ids):
        return [i2word[x] for x in list_ids]

    with gzip.open(path) as file:

        for json_game in file:
            game = json.loads(json_game.decode("utf-8"))

            if successful_only:
                if not game['status'] == 'success':
                    continue

            if len(game['qas']) > max_no_qs:
                continue

            # if freq_dial_len[len(game['qas'])] < 5000:
            #     freq_dial_len[len(game['qas'])] += 1
            # else:
            #     continue

            objects = list()
            object_ids = list() # These are added for crop features
            spatials = list()
            bboxes = list()
            target = int()
            target_cat = int()
            for i, o in enumerate(game['objects']):
                objects.append(o['category_id'])
                object_ids.append(o['id'])
                spatials.append(get_spatial_feat(bbox=o['bbox'], im_width=game['image']['width'], im_height=game['image']['height']))

                if o['id'] == game['object_id']:
                    target = i
                    target_cat = o['category_id']
                    bboxes.append(o['bbox'])

            for i, o in enumerate(game['objects']):
                if o['id'] == game['object_id']:
                    # target object information
                    spatial = get_spatial_feat(bbox=o['bbox'], im_width=game['image']['width'],
                                               im_height=game['image']['height'])
                    object_category = o['category_id']
                    break

            # pad objects, spatials and bboxes
            objects.extend([category_pad_token] * (max_no_objects - len(objects)))
            object_ids.extend([0] * (max_no_objects - len(object_ids)))
            spatials.extend([[0] * no_spatial_feat] * (max_no_objects - len(spatials)))

            # dialogue history and target question
            src = list()
            src_lengths = list()

            prev_hist = list()
            prev_answer = list()

            for i, qa in enumerate(game['qas']):

                if i != 0:
                    # remove padding from previous target and current source
                    src_unpad = src[:src.index(word2i['<padding>'])] if word2i['<padding>'] in src else src
                    target_q_unpad = target_q[:target_q.index(word2i['<padding>'])] if word2i['<padding>'] in target_q else target_q
                    src = src_unpad + target_q_unpad + [ans2tok[answer]]
                else:
                    src = [word2i[start]]
                src_lengths.append(len(src))

                answer = qa['answer']

                # oracle
                q_tokens = tknzr.tokenize(qa['question'])
                q_token_ids = [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens][:max_src_length]
                q_token_ids_start = [word2i['<start>']] + [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens][:max_src_length]
                a_token = ans2tok_oracle[qa['answer']]
                length = len(q_token_ids_start)
                if i == 0:
                    prev_hist = copy.deepcopy(q_token_ids_start)
                    prev_hist_len = len(q_token_ids_start)
                else:
                    prev_hist = prev_hist_copy + prev_answer + copy.deepcopy(q_token_ids)
                    prev_hist_len = p_h_l + 1 + len(q_token_ids)

                question = q_token_ids_start
                question_length = length

                prev_answer = [ans2tok[qa['answer']]]
                prev_hist_copy = copy.deepcopy(prev_hist)
                p_h_l = len(prev_hist_copy)

                prev_hist.extend([word2i['<padding>']] * (max_src_length - len(prev_hist)))
                question.extend([word2i['<padding>']] * (max_src_length - len(question)))

                # end oracle

                # if answer.lower().strip() == 'yes' and len(game['qas'])==1:
                #     ans_yes+=1
                # elif answer.lower().strip() == 'no' and len(game['qas'])==1:
                #     ans_no+=1
                # elif len(game['qas'])==1:
                #     ans_na+=1

                target_q = [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens]
                src_q = [word2i[start]] + [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens]

                # All decider targets here are 0
                n2n_data[_id] = dict()
                n2n_data[_id]['tgt_len'] = min(len(target_q), max_q_length)
                n2n_data[_id]['history_len'] = min(len(src), max_src_length)
                history_q_lens = src_lengths[:] # Deep copy
                len_hql = len(history_q_lens)
                history_q_lens.extend([0]*((max_no_qs+1)-len(history_q_lens))) # +1 because of start token is consiidered as first question
                n2n_data[_id]['history_q_lens'] = history_q_lens
                n2n_data[_id]['len_hql'] = len_hql
                target_q.extend([word2i['<padding>']] * (max_q_length - len(target_q)))
                src_q.extend([word2i['<padding>']] * (max_q_length - len(src_q)))
                src.extend([word2i['<padding>']] * (max_src_length - len(src)))
                n2n_data[_id]['history'] = src[:max_src_length]
                n2n_data[_id]['src_q'] = src_q[:max_q_length]
                n2n_data[_id]['target_q'] = target_q[:max_q_length]
                n2n_data[_id]['decider_tgt'] = 0
                decider_mask = [n2n_data[_id]['decider_tgt']]*len_hql
                decider_mask.extend([decidermask_pad_token]*((max_no_qs+1)-len_hql))
                n2n_data[_id]['decider_mask'] = decider_mask
                n2n_data[_id]['objects'] = objects
                n2n_data[_id]['object_ids'] = object_ids
                n2n_data[_id]['spatials'] = spatials
                n2n_data[_id]['target_obj'] = target
                n2n_data[_id]['target_cat'] = target_cat
                n2n_data[_id]['bboxes'] = bboxes # Change in v2 only target bbox is included as everything is not required
                n2n_data[_id]['game_id'] = str(game['id'])
                n2n_data[_id]['image_file'] = game['image']['file_name']
                n2n_data[_id]['image_url'] = game['image']['coco_url']

                n2n_data[_id]['oracle_question'] = question
                n2n_data[_id]['oracle_length'] = question_length
                n2n_data[_id]['oracle_prev_hist'] = prev_hist
                n2n_data[_id]['oracle_prev_hist_len'] = prev_hist_len
                n2n_data[_id]['oracle_answer'] = a_token
                n2n_data[_id]['oracle_spatial'] = spatial
                n2n_data[_id]['oracle_obj_cat'] = object_category
                # print(_id,n2n_data[_id])
                _id += 1

            src_unpad = src[:src.index(word2i['<padding>'])] if word2i['<padding>'] in src else src
            target_q_unpad = target_q[:target_q.index(word2i['<padding>'])] if word2i['<padding>'] in target_q else target_q
            src = src_unpad +  target_q_unpad + [ans2tok[answer]]
            src_q = [0]
            target_q = [0]
            src_lengths.append(len(src))

            # Decider target 1
            n2n_data[_id] = dict()
            n2n_data[_id]['tgt_len'] = min(len(target_q), max_q_length)
            n2n_data[_id]['history_len'] = min(len(src), max_src_length)
            history_q_lens = src_lengths[:] # Deep copy
            len_hql = len(history_q_lens)
            history_q_lens.extend([0]*((max_no_qs+1)-len(history_q_lens))) # +1 because of start token is consiidered as first question
            n2n_data[_id]['history_q_lens'] = history_q_lens
            n2n_data[_id]['len_hql'] = len_hql
            target_q.extend([word2i['<padding>']] * (max_q_length - len(target_q)))
            src_q.extend([word2i['<padding>']] * (max_q_length - len(src_q)))
            src.extend([word2i['<padding>']] * (max_src_length - len(src)))
            n2n_data[_id]['history'] = src[:max_src_length]
            n2n_data[_id]['src_q'] = src_q[:max_q_length]
            n2n_data[_id]['target_q'] = target_q[:max_q_length]
            n2n_data[_id]['decider_tgt'] = 1
            decider_mask = [0]*(len_hql-1) + [n2n_data[_id]['decider_tgt']]*1 # Because only the last target is guess
            decider_mask.extend([decidermask_pad_token]*((max_no_qs+1)-len_hql))
            n2n_data[_id]['decider_mask'] = decider_mask
            n2n_data[_id]['objects'] = objects
            n2n_data[_id]['object_ids'] = object_ids
            n2n_data[_id]['spatials'] = spatials
            n2n_data[_id]['target_obj'] = target
            n2n_data[_id]['target_cat'] = target_cat
            n2n_data[_id]['bboxes'] = bboxes  # Change in v2 only target bbox is included as everything is not required
            n2n_data[_id]['game_id'] = str(game['id'])
            n2n_data[_id]['image_file'] = game['image']['file_name']
            n2n_data[_id]['image_url'] = game['image']['coco_url']
            # print(_id, n2n_data[_id])

            n2n_data[_id]['oracle_question'] = question
            n2n_data[_id]['oracle_length'] = question_length
            n2n_data[_id]['oracle_prev_hist'] = prev_hist
            n2n_data[_id]['oracle_prev_hist_len'] = prev_hist_len
            n2n_data[_id]['oracle_answer'] = a_token
            n2n_data[_id]['oracle_spatial'] = spatial
            n2n_data[_id]['oracle_obj_cat'] = object_category

            _id += 1
            # break

    a=0
    # n2n_data_path = os.path.join(data_dir, data_file_name)
    n2n_data_path = os.path.join('/mnt/povobackup/clic/alberto.testoni/guesswhat-lxmert/data/', data_file_name)
    with open(n2n_data_path, 'w') as f:
        json.dump(n2n_data, f)
    print(freq_dial_len)
    #
    # print('Done')


if __name__ == '__main__':
    split = 'test'
    data_dir = 'data'
    data_file = "guesswhat.test.jsonl.gz"
    vocab_file = 'vocab.json'

    data_args = {
                'max_src_length' : 200,
                'max_q_length' : 30,
                'max_no_objects' : 20,
                'max_no_qs' : 10,
                'successful_only': True,
                'data_paths': ''
                }

    create_data_file(data_dir=data_dir, data_file=data_file, data_args=data_args, vocab_file_name=vocab_file, split=split)
