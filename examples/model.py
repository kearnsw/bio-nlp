import torch
from torch import nn
from torch import autograd
import argparse
from utils import idx_words, text2seq, idx_tags, tags2idx
import sys
from config import START_TAG, STOP_TAG

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_len, word2idx, tag2idx, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dims = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_len = vocab_len
        self.tag2idx = tag2idx
        self.word2idx = word2idx
        self.tagset_size = len(tag2idx)

        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

        # Condense VSM to tag space
        self.tags = nn.Linear(hidden_dim, len(tag2idx))

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag2idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.tags(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag2idx[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag2idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == "__main__":

    flags = argparse.ArgumentParser()
    flags.add_argument("--embeddingDims", type=int, help="size of word embeddings")
    flags.add_argument("--hiddenDims", type=int, help="size of hidden states")
    flags.add_argument("--epochs", type=int, help="number of training iterations")
    flags.add_argument("-l", "--learning_rate", type=float, help="rate to update gradients")
    flags.add_argument("-l2", "--regularization", type=float)
    args = flags.parse_args()

    """
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    """

    data = load_training_data()
    training_data = data[:100] 
    word2idx = idx_words(training_data)
    tag2idx = idx_tags(training_data)
    model = BiLSTM_CRF(len(word2idx), word2idx, tag2idx, len(word2idx), 64)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.001)

    for epoch in range(20):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        count = 0
        for sentence, tags in training_data:
            sys.stdout.write("Sentence {0}\n".format(count))
            sys.stdout.flush()
            model.zero_grad()

            sequence = text2seq(sentence, word2idx)
            tag_seq = torch.LongTensor([tag2idx[t] for t in tags])
            
            neg_log_likelihood = model.neg_log_likelihood(sequence, tag_seq)
            neg_log_likelihood.backward()
            optimizer.step()
            count += 1
    
    torch.save(model, "model.pkl")
    torch.save(model.state_dict(), "state.pkl")
    prediction = model(text2seq(data[200][0], word2idx))
    truth = tags2idx(data[200][1], tag2idx)
    print(truth.size())
    truth = truth.resize(1, truth.size()[0])
    print(truth)
    print(prediction)
    print(tag2idx)
