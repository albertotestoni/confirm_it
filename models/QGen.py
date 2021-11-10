import torch
import torch.nn as nn
import json
from utils.wrap_var import to_var
use_cuda = torch.cuda.is_available()


class QGenSeq2Seq(nn.Module):
    """
    QGen hidden state is initialised by the scaled encoder output.
    The input at every time step is the word embedding concatenated
    with the visual features scaled down to 512?
    """

    def __init__(self, **kwargs):
        super(QGenSeq2Seq, self).__init__()
        """
        Parameters
        ----------
        kwargs : dict
        'vocab_size' : vocabulary size
        'embedding_dim' : dimension of the word embeddings
        'word_pad_token' : Padding token in the vocab
        'num_layers' : Number of layers in the LSTM
        'hidden_dim' : Hidden state dimension
        'visual_features_dim' : Dimension of the visual features
        'scale_visual_to' : Dimension to reduce the visual features to.
        """

        # with open(os.path.join('data/vocab.json')) as in_file:
        #     self.vocab = json.load(in_file)["i2word"]
        #     print("vocab")

        self.qgen_args = kwargs

        self.word_embedding = nn.Embedding(
            self.qgen_args['vocab_size'],
            self.qgen_args['word_embedding_dim'],
            padding_idx=self.qgen_args['word_pad_token']
        )

        self.scale_visual_to = nn.Linear(self.qgen_args['visual_features_dim'], self.qgen_args['scale_visual_to'])

        self.rnn = nn.LSTM(
            self.qgen_args['word_embedding_dim'] + self.qgen_args['scale_visual_to'],
            self.qgen_args['hidden_dim'],
            num_layers=self.qgen_args['num_layers'],
            batch_first=True
        )

        # TODO: make it get_attr for option of GRU

        self.to_logits = nn.Linear(self.qgen_args['hidden_dim'], self.qgen_args['vocab_size'])

        self.start_token = self.qgen_args['start_token']
        self.startToken = self.qgen_args['start_token']
        self.endToken = 12
        self.padToken = 0
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.vocabSize =  self.qgen_args['vocab_size']
        self.logSoftmax = nn.LogSoftmax(dim=1)

    # def idx2sentence(self, tokens):
    #     s=[]
    #     for t in tokens.data:
    #         s.append(self.vocab[str(int(t))])
    #         if int(t)==12:
    #             break
    #     return s

    def forward(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
        'visual_features': 2048 dim avg pool layer of ResNet 152 [Bx2048]
        'src_q': The input word sequence during teacher forcing training part. Shape: [Bx max_q_length]
        'encoder_hidden': output from the encoder. Shape: [Bx1xhidden_size]
        'lengths': target length for masking to calculate the loss

        Returns
        -------
        output : dict
        'packed_word_logits': predicted words
        """

        src_q, encoder_hidden, visual_features = kwargs['src_q'], kwargs['encoder_hidden'], kwargs['visual_features']

        lengths = kwargs['lengths']

        batch_size = encoder_hidden.size(0)

        # concatenating encoder hidden and visual features and scaling to required QGen hidden size
        hidden = encoder_hidden.transpose(1, 0)
        proj_visual_features = self.ReLU(self.scale_visual_to(visual_features))

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))

        # copy visual features for each input token
        proj_visual_features_batch = proj_visual_features.repeat(1, self.qgen_args['max_tgt_length']).view(
            batch_size,
            self.qgen_args['max_tgt_length'],
            proj_visual_features.size(1)
        )

        # get word embeddings for input tokens
        src_q_embedding = self.word_embedding(src_q)

        input = torch.cat([src_q_embedding, proj_visual_features_batch], dim=2)

        # RNN forward pass
        # Aapparently hidden and cell have to be in num_layers x batch_size x hidden_dim
        rnn_hiddens, _ = self.rnn(input, (hidden, cell))

        rnn_hiddens.contiguous()

        word_logits = self.to_logits(rnn_hiddens.contiguous().view(-1, self.qgen_args['hidden_dim'])).contiguous().view(
            batch_size,
            self.qgen_args['max_tgt_length'],
            self.qgen_args['vocab_size']
        )

        # Packing them to act as an alternative to masking
        # packed_word_logits = pack_padded_sequence(word_logits, list(lengths.data), batch_first=True)[0]

        return word_logits

    # TODO Beam Search and Gumbel Smapler.

    def basicforward(self, embedding, rnn_state):
        """Short summary.

        Parameters
        ----------
        embedding :
        rnn_state :

        Returns
        -------
        logits:
        rnn_state:
        """
        rnn_hiddens, rnn_state = self.rnn(embedding, rnn_state)
        rnn_hiddens.contiguous()
        logits = self.to_logits(rnn_hiddens.view(-1, self.qgen_args['hidden_dim'])).view(embedding.size(0), 1,
                                                                                         self.qgen_args['vocab_size'])

        return logits, rnn_state

    def sampling(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict

        Returns
        -------

        """
        greedy = kwargs.get('greedy', False)
        return_logits_and_hidden_states = kwargs.get('return_logits_and_hidden_states', False)
        beam = kwargs.get('beam_size', 1)

        if not greedy:
            temp = kwargs.get('temp', 1)

        encoder_hidden, visual_features = kwargs['encoder_hidden'], kwargs['visual_features']

        batch_size = encoder_hidden.size(0)

        start_tokens = to_var(torch.LongTensor(batch_size, 1).fill_(self.start_token))

        hidden = encoder_hidden.transpose(1, 0)
        proj_visual_features = self.ReLU(self.scale_visual_to(visual_features)).unsqueeze(1)

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))

        start_embedding = self.word_embedding(start_tokens)

        _embedding = torch.cat([start_embedding, proj_visual_features], dim=2)

        rnn_state = (hidden, cell)

        sampled_q_logits = []
        sampled_q_tokens = []
        decoder_hidden_states = []
        for i in range(self.qgen_args['max_tgt_length']):
            if i > 0:
                word_embedding = self.word_embedding(word_id)
                _embedding = torch.cat([word_embedding, proj_visual_features], dim=2)
            decoder_hidden_states.append(rnn_state[0].squeeze(0))
            logits, rnn_state = self.basicforward(embedding=_embedding, rnn_state=rnn_state)

            if greedy:
                word_prob, word_id = self.softmax(logits).max(-1)
            else:
                # Make sure that temp is between 0.1-1 for good results. temp=0 will throw an error.
                probabilities = self.softmax(logits / temp).squeeze()
                m = torch.distributions.Categorical(probabilities)
                tmp_token = m.sample()
                word_prob = m.log_prob(tmp_token).view(batch_size, 1)
                word_id = tmp_token.long().view(batch_size, 1)

            sampled_q_logits.append(word_prob)
            sampled_q_tokens.append(word_id)

        sampled_q_logits = torch.cat(sampled_q_logits, dim=1)
        sampled_q_tokens = torch.cat(sampled_q_tokens, dim=1)

        if greedy:
            sampled_q_logits = torch.log(sampled_q_logits)

        if return_logits_and_hidden_states:
            return sampled_q_tokens, sampled_q_logits, decoder_hidden_states
        else:
            return sampled_q_tokens

    def beamSearchDecoder(self, recover_top=True, **kwargs):
        '''
        Beam search for sequence generation
        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        '''

        beamSize = kwargs.get('beam_size', 1)

        # if beamSize == 1:
        #     print("++++GREEDY++++")

        encoder_hidden, visual_features = kwargs['encoder_hidden'], kwargs['visual_features']
        maxLen = self.qgen_args['max_tgt_length']
        batchSize = encoder_hidden.size(0)

        start_tokens = to_var(torch.LongTensor(batchSize, 1).fill_(self.start_token))

        hidden = encoder_hidden.transpose(1, 0)
        proj_visual_features = self.ReLU(self.scale_visual_to(visual_features)).unsqueeze(1)

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batchSize, self.qgen_args['hidden_dim']))

        start_embedding = self.word_embedding(start_tokens)

        _embedding = torch.cat([start_embedding, proj_visual_features], dim=2)

        hiddenStates = (hidden, cell)

        if use_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        startTokenArray = to_var(startTokenArray)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)

        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = to_var(tokenArange)

        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(self.endToken)
        beamTokensTable = to_var(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = to_var(backIndices)

        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

        for t in range(self.qgen_args['max_tgt_length']):
            if t == 0:
                logits, hiddenStates = self.basicforward(embedding=_embedding, rnn_state=hiddenStates)
                logProbs = self.logSoftmax(logits.squeeze(1))
                # Find top beamSize logProbs
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                logProbSums = topLogProbs

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(1, -1, 512)
                    for x in hiddenStates
                ]
                a=0
                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                # proj_visual_features_batch = proj_visual_features.repeat(beamSize,1,1)
                proj_visual_features_batch = proj_visual_features.repeat_interleave(beamSize, 1).view(
                    proj_visual_features.size(0) * beamSize, 1,
                    proj_visual_features.size(2))
                word_embedding = self.word_embedding(beamTokensTable[:, :, t - 1])
                word_embedding = word_embedding.view(-1, 1, self.qgen_args['hidden_dim'])
                _embedding = torch.cat([word_embedding, proj_visual_features_batch], dim=2)
                # emb has shape (batchSize, beamSize, embedSize)
                logits, hiddenStates = self.basicforward(embedding=_embedding, rnn_state=hiddenStates)
                # output, hiddenStates = self.rnn(
                #     emb.view(-1, 1, self.qgen_args['hidden_dim']), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                # scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(logits.squeeze(1))
                # logProbs has shape (batchSize*beamSize, vocabSize)
                # NOTE: Padding token has been removed from generator output during
                # sampling (RL fine-tuning). However, the padding token is still
                # present in the generator vocab and needs to be handled in this
                # beam search function. This will be supported in a future release.
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                       self.vocabSize)

                if LENGTH_NORM:
                    # Add (current log probs / (t+1))
                    logProbs = logProbsCurrent * (aliveVector.float() /
                                                  (t + 1))
                    # Add (previous log probs * (t/t+1) ) <- Mean update
                    coeff_ = aliveVector.eq(0).float() + (
                            aliveVector.float() * t / (t + 1))
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    # Add currrent token logProbs for alive beams only
                    logProbs = logProbsCurrent * (aliveVector.float())
                    # Add previous logProbSums upto t-1
                    logProbs += logProbSums.unsqueeze(2)

                # Masking out along |V| dimension those sequence logProbs
                # which correspond to ended beams so as to only compare
                # one copy when sorting logProbs
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :,
                0] = 0  # Zeroing all except first row for ended beams
                minus_infinity_ = torch.min(logProbs).item()
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0). \
                    repeat(batchSize, beamSize, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2). \
                    repeat(1, 1, self.vocabSize).view(batchSize, -1)

                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                logProbSums = topLogProbs
                beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Update corresponding hidden and cell states for next time step
                hiddenCurrent, cellCurrent = hiddenStates

                # Reshape to get explicit beamSize dim
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)

                # Update states according to the next top beams
                backIndexVector = backIndices[:, :, t].unsqueeze(0) \
                    .unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)

                # Restore original shape for next rnn forward
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = (hiddenCurrent, cellCurrent)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        RECOVER_TOP_BEAM_ONLY = recover_top

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:, :, tokenIdx]. \
                          gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx]. \
                gather(1, backID)
            tokenIdx = tokenIdx - 1

        # tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)
        # idx_game = 19
        # for b in range(beamSize):
        #     print(' '.join(self.idx2sentence(tokens=tokens[idx_game][b])), "(", round(logProbSums[idx_game][b].item(),3), ")")
        # print("----")

        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens_n = tokens[:, 0]  # Keep only top beam
            seqLens = seqLens[:, 0]

            return to_var(tokens_n)
        else:
            return to_var(tokens)
