import torch.nn as nn

from models.Decider import Decider
from models.Encoder import Encoder
from models.Guesser import Guesser
from models.QGen import QGenSeq2Seq
from models.QGenImgCap import QGenImgCap
from models.Oracle_multitask import Oracle
import json

"""
Putting all the models together
"""


class Ensemble(nn.Module):
    """docstring for Ensemble."""

    def __init__(self, **kwargs):
        super(Ensemble, self).__init__()
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'encoder' : Arguments for the encoder module
            'qgen' : Arguments for the qgen module
            'guesser' : Arguments for the guesser module
            'regressor' : Arguments for the regressor module
            'decider' : Arguments for the decider module

        """

        self.ensemble_args = kwargs

        # TODO: use get_attr to get different versions of the same model. For example QGen

        self.encoder = Encoder(**self.ensemble_args['encoder'])

        # Qgen selection
        # For the NAACL 2019, we used Seq2Seq one
        if self.ensemble_args['qgen']['qgen'] == 'qgen_cap':
            self.qgen = QGenImgCap(**self.ensemble_args['qgen'])
        else:
            self.qgen = QGenSeq2Seq(**self.ensemble_args['qgen'])

        self.guesser = Guesser(**self.ensemble_args['guesser'])

        with open('config/Oracle/config.json') as in_file:
            oracle_config = json.load(in_file)

        self.oracle = Oracle(
            no_words=4901,
            no_words_feat=oracle_config['embeddings']['no_words_feat'],
            no_categories=oracle_config['embeddings']['no_categories'],
            no_category_feat=oracle_config['embeddings']['no_category_feat'],
            no_hidden_encoder=oracle_config['lstm']['no_hidden_encoder'],
            mlp_layer_sizes=oracle_config['mlp']['layer_sizes'],
            no_visual_feat=oracle_config['inputs']['no_visual_feat'],
            no_crop_feat=oracle_config['inputs']['no_crop_feat'],
            dropout=oracle_config['lstm']['dropout'],
            inputs_config=oracle_config['inputs'],
            scale_visual_to=oracle_config['inputs']['scale_visual_to']
        )

        self.decider = Decider(**self.ensemble_args['decider'])

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'history' : The dialogue history. Shape :[Bx max_src_length]
            'history_len' : The length of the dialogue history. Shape [Bx1]
            'src_q' : The input word sequence for the QGen
            'tgt_len' : Length of the target question
            'visual_features' : The avg pool layer from ResNet 152
            'spatials' : Spatial features for the guesser. Shape [Bx20x8]
            'objects' : List of objects for guesser. Shape [Bx20]
            'mask_select' : Bool. Based on the decider target, either QGen or Guesser is used

        Returns
        -------
        ensemble_out : dict
            'decider_out' : predicted decision
            'guesser_out' : log probabilities of the objects
            'qgen_out' : predicted next question

        """
        # history, history_len = kwargs['history'], kwargs['history_len']
        history, history_len = kwargs.get('history', None), kwargs.get('history_len', None)
        lengths = kwargs.get('tgt_len', None)
        visual_features = self.dropout(kwargs['visual_features'])
        src_q = kwargs.get('src_q', None)
        spatials = kwargs.get('spatials', None)
        objects = kwargs.get('objects', None)
        mask_select = kwargs['mask_select']

        objects_feat = kwargs.get("objects_feat", None)
        if mask_select != 2:
            encoder_hidden = self.encoder(history=history, history_len=history_len, visual_features=visual_features)
            self.gdse = encoder_hidden
            decider_out = self.decider(encoder_hidden=encoder_hidden)

        if mask_select == 1:
            guesser_out = self.guesser(encoder_hidden=encoder_hidden, spatials=spatials, objects=objects,
                                       objects_feat=objects_feat, regress=False)

            return decider_out, guesser_out
        elif mask_select == 0:
            qgen_out = self.qgen(src_q=src_q, encoder_hidden=encoder_hidden, visual_features=visual_features,
                                 lengths=lengths)

            return decider_out, qgen_out

        elif mask_select == 2:

            encoder_hidden = self.encoder(history=kwargs['oracle_prev_hist'],
                                          history_len=kwargs['oracle_prev_hist_len'],
                                          visual_features=visual_features)
            self.gdse = encoder_hidden
            decider_out = self.decider(encoder_hidden=encoder_hidden)

            # oracle_question = kwargs['oracle_question']
            # oracle_length = kwargs['oracle_length']
            # oracle_prev_hist = kwargs['oracle_prev_hist']
            # oracle_prev_hist_len = kwargs['oracle_prev_hist_len']
            oracle_spatial = kwargs['oracle_spatial']
            oracle_obj_cat = kwargs['oracle_obj_cat']

            oracle_out = self.oracle(questions=None, lengths=None, obj_categories=oracle_obj_cat,
                                     spatials=oracle_spatial, encoder_hidden=encoder_hidden, crop_features=None, visual_features=None)

            return decider_out, oracle_out
        else:
            print("error")
            exit(0)
