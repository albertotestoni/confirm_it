import datetime
import json
from time import time

import os
import torch

from utils.config import load_config

# from utils.vocab import create_vocab

use_cuda = torch.cuda.is_available()

def preprocess_config(args):
    """Function to process the arguments and returns the relevant arguments for training.

    Parameters
    ----------
    args : type
        check config.json for all the dictionary keys

    Returns
    -------
    ensemble_args : dict. Arguments for all the modules.
    dataset_args : dict. Arguments for Dataset
    optimizer_args : dict. Arghuments for the optimizer
    exp_config : dict. Arguments for saving models, logging etc...

    """
    # TODO Also load arguments for visualisation

    config = load_config(args.config)
    ens_config = load_config(args.ens_config)
    or_config = load_config(args.or_config)

    # Create vocab.json
    if config['dataset']['new_vocab'] or not os.path.isfile(os.path.join(args.data_dir, config['data_paths']['vocab_file'])):
        create_vocab(
            data_dir=args.data_dir,
            data_file=config['data_paths']['train'],
            min_occ=config['dataset']['min_occ'])

    with open(os.path.join(args.data_dir, config['data_paths']['vocab_file'])) as file:
        vocab = json.load(file)
    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)
    word_pad_token = word2i['<padding>']

    ensemble_args = dict()

    # Experiments_args
    exp_config = config['exp_config']
    exp_config['ts'] = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))

    # Encoder args
    ensemble_args['encoder'] = ens_config['encoder']
    ensemble_args['encoder']['vocab_size'] = vocab_size
    ensemble_args['encoder']['word_embedding_dim'] = ens_config['embeddings']['word_embedding_dim']
    ensemble_args['encoder']['word_pad_token'] = word_pad_token
    # For gameplay regardless of the decider, the forward pass for LSTM decider is similar to MLP one
    ensemble_args['encoder']['decider'] = 'decider'

    # Guesser args
    ensemble_args['guesser'] = ens_config['guesser']
    ensemble_args['guesser']['no_categories'] = ens_config['embeddings']['no_categories']
    ensemble_args['guesser']['obj_pad_token'] = ens_config['embeddings']['obj_pad_token']
    ensemble_args['guesser']['obj_categories_embedding_dim'] = ens_config['embeddings']['obj_categories_embedding_dim']
    # ensemble_args['guesser']['encoder_hidden_dim'] = config['encoder']['hidden_dim']


    # QGen args
    if exp_config['qgen'] == 'qgen_cap':
        ensemble_args['qgen'] = ens_config['qgen_cap']
        ensemble_args['qgen']['qgen'] = 'qgen_cap'
        ensemble_args['qgen']['encoder_hidden_dim'] = ens_config['encoder']['scale_to']
    else:
        ensemble_args['qgen'] = ens_config['qgen']
        ensemble_args['qgen']['qgen'] = 'qgen'
    ensemble_args['qgen']['max_tgt_length'] = config['dataset']['max_q_length']
    ensemble_args['qgen']['vocab_size'] = vocab_size
    ensemble_args['qgen']['word_embedding_dim'] = ens_config['embeddings']['word_embedding_dim']
    ensemble_args['qgen']['word_pad_token'] = word_pad_token
    ensemble_args['qgen']['visual_features_dim'] = ens_config['encoder']['visual_features_dim']
    ensemble_args['qgen']['start_token'] = word2i['<start>']

    # Decider
    ensemble_args['decider'] = ens_config['decider']
    ensemble_args['decider']['type'] = config['exp_config']['decider']

    # Dataset
    dataset_args = config['dataset']
    dataset_args['data_dir'] = args.data_dir
    dataset_args['data_paths'] = config['data_paths']

    # ADD
    dataset_args['breaking'] = args.breaking

    if "FasterRCNN" in config['data_paths']:
        dataset_args["FasterRCNN"] = config['data_paths']["FasterRCNN"]

    # Optimizer_args
    optimizer_args = config['optimizer']
    optimizer_args['my_cpu'] = args.my_cpu
    if args.batch_size != None:
        optimizer_args['batch_size'] = args.batch_size

    oracle_args = or_config['oracle']
    oracle_args['vocab_size'] = vocab_size

    ensemble_args['bin_file'] = ens_config['ensemble']['bin_file']
    if args.load_bin_path != None:
        ensemble_args['bin_file'] = args.load_bin_path

    ensemble_args['encoder']['decider'] = 'decider'

    with open(os.path.join(args.data_dir, config['data_paths']['catid2str'])) as file:
        catid2str = json.load(file)

    return ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i, i2word, catid2str
