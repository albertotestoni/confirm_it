import argparse
import json
from time import time

import numpy as np
import os
import torch.nn as nn
import tqdm
from shutil import copy2
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from models.CNN import ResNet
from models.Ensemble_Confirm_it_3tasks import Ensemble
from train.GamePlay.parser import preprocess_config
from utils.datasets.GamePlay.GamePlayDataset import GamePlayDataset
from utils.eval import calculate_accuracy_all
from utils.gameplayutils import *
from utils.model_loading import load_model

USE_CUDA = torch.cuda.is_available()


def append_dialogue_no_answers(dialogue, dialogue_length, new_questions, question_length, pad_token):
    """
    Given a dialogue history, will append a new question to it.
    Will take care of padding, possible cutting off (if required) the dialogue as well as
    returning updated dialogue length.
    """

    max_dialogue_length = dialogue.size(1)

    for qi, q in enumerate(new_questions):
        # put new dialogue together from old dialogue + new question + answer
        updated_dialogue = torch.cat(
            [
                dialogue[qi][:dialogue_length[qi].item()],
                new_questions[qi, :question_length.data[qi]]
            ]
        )

        # update length
        dialogue_length[qi] = min(updated_dialogue.size(0), max_dialogue_length)

        # strip and pad
        updated_dialogue = updated_dialogue[:max_dialogue_length]
        if updated_dialogue.size(0) < dialogue.size(1):
            dialogue_pad = to_var(torch.Tensor(max_dialogue_length - updated_dialogue.size(0)).fill_(pad_token).long())
            updated_dialogue = torch.cat([updated_dialogue, dialogue_pad])

        dialogue[qi] = updated_dialogue.unsqueeze(0)

    return dialogue, dialogue_length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="../guesswhat-lxmert/data", help='Data Directory')
    parser.add_argument("-config", type=str, default="../guesswhat-lxmert/config/GamePlay/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="../guesswhat-lxmert/config/GamePlay/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="../guesswhat-lxmert/config/GamePlay/oracle.json", help=' Oracle config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which were saved with Dataparallel')
    parser.add_argument("-log_enchidden", action='store_true', help='This flag saves the encoder hidden state. WARNING!!! This might cause the resulting json file to blow up!')
    parser.add_argument("-batch_size", type=int, help='Batch size for the gameplay')
    parser.add_argument("-load_bin_path", type=str, help='Bin file path for the saved model. If this is not given then one provided in ensemble.json will be taken ')

    args = parser.parse_args()
    print(args.exp_name)
    use_dataparallel = args.dataparallel
    breaking = args.breaking

    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i, i2word, catid2str = preprocess_config(args)

    pad_token= word2i['<padding>']

    torch.manual_seed(exp_config['seed'])
    if USE_CUDA:
        torch.cuda.manual_seed_all(exp_config['seed'])

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args)))  # converting args.namespace to dict

    if args.resnet:
        cnn = ResNet()

        if USE_CUDA:
            cnn.cuda()
            cnn = DataParallel(cnn)
        cnn.eval()

    softmax = nn.Softmax(dim=-1)

    dataset_test = GamePlayDataset(split='test', **dataset_args)

    dataset_args['max_no_qs'] = 5
    hypothesis_change = []
    for beam_size in [3]:
        print("BEAM SIZE: ", beam_size)
        with torch.no_grad():
            #  Always compare models trained for the same number of epochs (see Testoni and Bernardi, EACL 2021)
            for epoch in range(28,29):

                print("Epoch: {}".format(epoch))
                model_filename = args.load_bin_path+str(epoch)
                print("Loading model".format(model_filename))
                model = Ensemble(**ensemble_args)
                model = load_model(model, model_filename, use_dataparallel=use_dataparallel)
                model.eval()

                for split, dataset in zip(["test"], [dataset_test]):
                    print(split)
                    eval_log = dict()

                    dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=optimizer_args['batch_size'],
                    shuffle=False,
                    num_workers= 0,
                    pin_memory= USE_CUDA,
                    drop_last=False)

                    total_no_batches = len(dataloader)
                    accuracy = list()
                    decider_perc = list()
                    start = time()

                    for i_batch, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):

                        # Get Batch
                        for k, v in sample.items():
                            if torch.is_tensor(v):
                                sample[k] = to_var(v, True)

                        if args.resnet:
                            img_features, avg_img_features = cnn(to_var(sample['image'].data, True))
                        else:
                            avg_img_features = sample['image']

                        batch_size = avg_img_features.size(0)

                        history = to_var(torch.LongTensor(batch_size, 200).fill_(pad_token))
                        history[:,0] = sample['history'].squeeze()
                        history_len = sample['history_len']

                        decisions = to_var(torch.LongTensor(batch_size).fill_(0))
                        mask_ind = torch.nonzero(1-decisions).squeeze()
                        _enc_mask = mask_ind

                        if exp_config['logging']:
                            decision_probs = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1, 1)))
                            all_guesser_probs = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1,20)))
                            if args.log_enchidden:
                                enc_hidden_logging = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1, ensemble_args['encoder']['scale_to'])))

                        for q_idx in range(dataset_args['max_no_qs']):

                            if use_dataparallel and USE_CUDA:
                                encoder_hidden = model.module.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                                decision = model.module.decider(encoder_hidden=encoder_hidden)
                            else:
                                encoder_hidden = model.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                                decision = model.decider(encoder_hidden=encoder_hidden)

                            ########## Logging Block ################
                            if exp_config['logging'] and q_idx==0:
                                if use_dataparallel and USE_CUDA:
                                    tmp_guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
                                else:
                                    tmp_guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

                                tmp_guesser_prob = softmax(tmp_guesser_logits * sample['objects_mask'].float())

                                for guesser_i, i in enumerate(mask_ind.data.tolist()):
                                    all_guesser_probs[i, q_idx, :] = tmp_guesser_prob[guesser_i]

                            if args.log_enchidden and exp_config['logging']:
                                for enc_i, i in enumerate(mask_ind.data.tolist()):
                                    enc_hidden_logging[i, q_idx, :] = encoder_hidden[enc_i]
                            ##########################################

                            if exp_config['decider_enabled']:
                                _decision = softmax(decision).max(-1)[1].squeeze()
                            else:
                                _decision = to_var(torch.LongTensor(decision.size(0)).fill_(0))

                            ########## Logging Block ################
                            if exp_config['logging'] and exp_config['decider_enabled']:
                                _decision_probs = softmax(decision)[:,:,1]
                                for dec_i, i in enumerate(mask_ind.data.tolist()):
                                    decision_probs[i, q_idx, :] = _decision_probs[dec_i]
                            ##########################################

                            decisions[mask_ind] = _decision
                            _enc_mask = torch.nonzero(1-_decision).squeeze()
                            mask_ind = torch.nonzero(1-decisions).squeeze()

                            if len(mask_ind)==0:
                                break

                            if q_idx not in [0]:

                                #  Let the Guesser assign probabilities to the candidate objects given the dialogue
                                #  history from the previous turn.
                                encoder_hidden_previous_turn = model.encoder(history=history,
                                                                             visual_features=avg_img_features,
                                                                             history_len=history_len)

                                guesser_logits_previous_turn = model.guesser(encoder_hidden=encoder_hidden_previous_turn,
                                                                             spatials=sample['spatials'],
                                                                             objects=sample['objects'],
                                                                             regress=False)

                                #  Get the candidates that receives the highest probability
                                top_obj = guesser_logits_previous_turn.topk(1)[1].squeeze().tolist()
                                top_probs = torch.max(softmax(guesser_logits_previous_turn * sample['objects_mask'].float()), dim=1)[0].detach().cpu().tolist()

                                #  Let the QGen generate a set of candidate follow-up questions via beam search
                                qgen_out = model.qgen.beamSearchDecoder(src_q=sample['src_q'][mask_ind],
                                                                        encoder_hidden=encoder_hidden[_enc_mask],
                                                                        visual_features=avg_img_features[mask_ind],
                                                                        recover_top=False, beam_size=beam_size)

                                qgen_out_beam = qgen_out.clone()
                                #  reshape the matrix and get questions' length
                                qgen_out = qgen_out.view(qgen_out.shape[0] * qgen_out.shape[1], qgen_out.shape[2])
                                new_question_lengths_beam = get_newq_lengths(qgen_out, word2i["?"])

                                #  append each new question to the dialogue history
                                history_beam, history_len_beam = append_dialogue_no_answers(
                                    dialogue=history.repeat_interleave(beam_size, 0),
                                    dialogue_length=history_len.repeat_interleave(beam_size, 0),
                                    new_questions=qgen_out,
                                    question_length=new_question_lengths_beam,
                                    pad_token=pad_token)

                                #  prepare the information about the target hypothesis for the Internal Oracle
                                idx_target_h = 0
                                spatials_target_h = torch.FloatTensor(qgen_out.shape[0], 8).fill_(0).cuda()
                                categories_target_h = torch.LongTensor(qgen_out.shape[0]).fill_(0).cuda()
                                for b_idx in range(batch_size):
                                    for ia in range(beam_size):
                                        spatials_target_h[idx_target_h] = sample['spatials'][b_idx][top_obj[b_idx]]
                                        categories_target_h[idx_target_h] = sample['objects'][b_idx][top_obj[b_idx]]
                                        idx_target_h += 1

                                # get the answers from the Interal Oracle given a target hypothesis
                                encoder_hidden_internal_o = model.encoder(history=history_beam,
                                                                          visual_features=avg_img_features.repeat_interleave(beam_size,0),
                                                                          history_len=history_len_beam)

                                answer_predictions_external_o = model.oracle(questions=None,
                                                                             lengths=None,
                                                                             obj_categories=categories_target_h,
                                                                             spatials=spatials_target_h,
                                                                             encoder_hidden=encoder_hidden_internal_o,
                                                                             crop_features=None,
                                                                             visual_features=None)

                                answer_tokens_internal_o = anspred2wordtok(answer_predictions_external_o, word2i)

                                #  compute the new probabilities and compare them with the ones from the previous turn
                                #  to re-rank the set of hypothesis questions
                                history_with_IO_answers, history_len_with_IO_answers = append_dialogue(
                                    dialogue=history.repeat_interleave(beam_size, 0),
                                    dialogue_length=history_len.repeat_interleave(beam_size, 0),
                                    new_questions=qgen_out,
                                    question_length=new_question_lengths_beam,
                                    answer_tokens=answer_tokens_internal_o,
                                    pad_token=pad_token)

                                encoder_hidden_guesser = model.encoder(history=history_with_IO_answers,
                                                                       visual_features=avg_img_features.repeat_interleave(beam_size,0),
                                                                       history_len=history_len_with_IO_answers)

                                guesser_logits_new_probs = model.guesser(encoder_hidden=encoder_hidden_guesser,
                                                                         spatials=sample['spatials'].repeat_interleave(beam_size,0),
                                                                         objects=sample['objects'].repeat_interleave(beam_size,0),
                                                                         regress=False)

                                guesser_logits_new_probs = softmax(guesser_logits_new_probs * sample['objects_mask'].repeat_interleave(beam_size,0).float()).view(batch_size, beam_size, guesser_logits_new_probs.shape[-1])
                                selected_q_idx = []
                                ans_a = answer_tokens_internal_o.squeeze().view(batch_size, beam_size)

                                #  for each game, get the index of the question that increases the most the probability
                                #  on the target hypothesis
                                for batch_id, beam_res in enumerate(guesser_logits_new_probs):
                                    sorted_idx = torch.argsort(beam_res[:,top_obj[batch_id]] - top_probs[batch_id], descending=True).tolist()
                                    selected_q_idx.append(sorted_idx[0])

                                #  get only the selected questions and move forward towards the External Oracle
                                qgen_out = torch.LongTensor(batch_size, qgen_out_beam.shape[-1]).fill_(pad_token).cuda()
                                for iif, q_q_idx in enumerate(selected_q_idx):
                                    qgen_out[iif] = qgen_out_beam[iif][q_q_idx]
                                new_question_lengths = get_newq_lengths(qgen_out, word2i["?"])

                                history_external_oracle, history_len_external_oracle = append_dialogue_no_answers(
                                    dialogue=history.clone(),
                                    dialogue_length=history_len.clone(),
                                    new_questions=qgen_out.clone(),
                                    question_length=new_question_lengths.clone(),
                                    pad_token=pad_token)

                                #  the external oracle (based on the real target) answers the questions that are
                                #  expected to help the most in confirming the model's conjecture about the target
                                encoder_hidden_external_o = model.encoder(history=history_external_oracle,
                                                                          visual_features=avg_img_features,
                                                                          history_len=history_len_external_oracle)

                                answer_predictions_external_o = model.oracle(questions=None,
                                                                             lengths=None,
                                                                             obj_categories=sample['target_cat'][mask_ind],
                                                                             spatials=sample['target_spatials'][mask_ind],
                                                                             encoder_hidden=encoder_hidden_external_o,
                                                                             crop_features=None,
                                                                             visual_features=None)

                                answer_tokens_external_o = anspred2wordtok(answer_predictions_external_o, word2i)

                                history[mask_ind], history_len[mask_ind] = append_dialogue(
                                    dialogue=history[mask_ind],
                                    dialogue_length=history_len[mask_ind],
                                    new_questions=qgen_out,
                                    question_length=new_question_lengths,
                                    answer_tokens=answer_tokens_external_o,
                                    pad_token= pad_token)

                            else:
                                #  during the first dialogue turn, generate a question via usual beam search
                                if use_dataparallel and USE_CUDA:
                                    qgen_out = model.module.qgen.beamSearchDecoder(src_q=sample['src_q'][mask_ind],
                                                                                   encoder_hidden=encoder_hidden[
                                                                                       _enc_mask],
                                                                                   visual_features=avg_img_features[
                                                                                       mask_ind], greedy=True,
                                                                                   beam_size=beam_size)
                                else:
                                    qgen_out = model.qgen.beamSearchDecoder(src_q=sample['src_q'][mask_ind],
                                                                            encoder_hidden=encoder_hidden[_enc_mask],
                                                                            visual_features=avg_img_features[mask_ind],
                                                                            greedy=True, beam_size=beam_size)


                                new_question_lengths = get_newq_lengths(qgen_out, word2i["?"])

                                history_external_oracle, history_len_external_oracle = append_dialogue_no_answers(
                                    dialogue=history.clone(),
                                    dialogue_length=history_len.clone(),
                                    new_questions=qgen_out.clone(),
                                    question_length=new_question_lengths.clone(),
                                    pad_token=pad_token)

                                encoder_hidden_external_o = model.encoder(history=history_external_oracle,
                                                                          visual_features=avg_img_features,
                                                                          history_len=history_len_external_oracle)

                                answer_predictions_external_o = model.oracle(questions=None,
                                                                             lengths=None,
                                                                             obj_categories=sample['target_cat'][mask_ind],
                                                                             spatials=sample['target_spatials'][mask_ind],
                                                                             encoder_hidden=encoder_hidden_external_o,
                                                                             crop_features=None,
                                                                             visual_features=None)

                                answer_tokens_external_o = anspred2wordtok(answer_predictions_external_o, word2i)

                                history[mask_ind], history_len[mask_ind] = append_dialogue(
                                    dialogue=history[mask_ind],
                                    dialogue_length=history_len[mask_ind],
                                    new_questions=qgen_out,
                                    question_length=new_question_lengths,
                                    answer_tokens=answer_tokens_external_o,
                                    pad_token=pad_token)

                            if dataset_args['max_no_qs']-1 == q_idx:
                                #  when the maximum number of dialogue turns is reached, guess the target object

                                if exp_config['decider_enabled']:
                                    if use_dataparallel and USE_CUDA:
                                        encoder_hidden = model.module.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                                        decision = model.module.decider(encoder_hidden=encoder_hidden)
                                    else:
                                        encoder_hidden = model.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                                        decision = model.decider(encoder_hidden=encoder_hidden)
                                    _decision = softmax(decision).max(-1)[1].squeeze()
                                    decisions[mask_ind] = _decision

                                    ########## Logging Block ################
                                    _decision_probs = softmax(decision)[:,:,1]
                                    if exp_config['logging']:
                                        for dec_i, i in enumerate(mask_ind.data.tolist()):
                                            decision_probs[i, q_idx+1, :] = _decision_probs[dec_i]

                                if args.log_enchidden and exp_config['logging']:
                                    for enc_i, i in enumerate(mask_ind.data.tolist()):
                                        enc_hidden_logging[i, q_idx, :] = encoder_hidden[enc_i]
                                    ##########################################

                            ########## Logging Block ################
                            if exp_config['logging']:
                                if use_dataparallel and USE_CUDA:
                                    encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                                    tmp_guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
                                else:
                                    encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                                    tmp_guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

                                tmp_guesser_prob = softmax(tmp_guesser_logits * sample['objects_mask'].float())

                                if exp_config['logging']:
                                    for guesser_i, i in enumerate(mask_ind.data.tolist()):
                                        all_guesser_probs[i, q_idx+1, :] = tmp_guesser_prob[guesser_i]
                            ##########################################

                        if use_dataparallel and USE_CUDA:
                            encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                            guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
                        else:
                            encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                            guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

                        batch_accuracy = calculate_accuracy_all(softmax(guesser_logits*sample['objects_mask'].float()), sample['target_obj'])
                        accuracy.extend(batch_accuracy)
                        decider_perc.append((torch.sum(decisions.data)/decisions.size(0)).item())

                    ########## Logging Block ################
                        if exp_config['logging']:

                            dials = dialtok2dial(history, i2word)
                            guesser_probs = softmax(guesser_logits*sample['objects_mask'].float())
                            guesses = guesser_probs.max(-1)[1]

                            for bidx in range(batch_size):
                                eval_log[sample['game_id'][bidx]] = dict()
                                eval_log[sample['game_id'][bidx]]['split'] = split
                                eval_log[sample['game_id'][bidx]]['gen_dialogue'] = dials[bidx]
                                eval_log[sample['game_id'][bidx]]['image'] = sample['image_file'][bidx]
                                eval_log[sample['game_id'][bidx]]['flickr_url'] = sample['image_url'][bidx]
                                eval_log[sample['game_id'][bidx]]['target_id'] = sample['target_obj'][bidx].item()
                                eval_log[sample['game_id'][bidx]]['guessed'] = int(batch_accuracy[bidx])
                                eval_log[sample['game_id'][bidx]]['all_guess_probs'] = all_guesser_probs[bidx].data.tolist()

                    if exp_config['logging']:
                        file_name = log_dir+split+ '_E_'+str(epoch)+ '_GPinference_'+str(args.exp_name)+'_'+exp_config['ts']+'.json'
                        with open(file_name, 'w') as f:
                            json.dump(eval_log, f)
                        print("Log file saved in: {}".format(file_name))
                    ##########################################

                    print("{} - {}".format(split, np.mean(accuracy)))

