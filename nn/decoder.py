import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

from .attention import build_attention

def build_decoder(num_layers, out_features, lstm_features, bert_hsize, bert_tokenizer, max_len, device='cuda:0'):
    decoder = Decoder(
        num_layers,
        out_features,
        lstm_features,
        bert_hsize,
        bert_tokenizer,
        max_len,
        device
    )

    return decoder

VOCAB_SIZE = 32006

class Decoder(nn.Module):
    def __init__(self, num_layers, out_features, lstm_features, bert_hsize, bert_tokenizer, max_len, device='cuda:0'):
        super(Decoder, self).__init__()

        self.bert_tokenizer = bert_tokenizer

        # Parameters ######################################
        self.num_layers = num_layers
        self.out_features = out_features
        self.lstm_features = lstm_features
        self.bert_hsize = bert_hsize
        self.device = device
        self.max_len = max_len
        ###################################################

        # Layers ##########################################
        self.embedding = nn.Embedding(VOCAB_SIZE, self.out_features)
        self.dense1 = nn.Linear(self.out_features, self.out_features)
        self.layer_norm0 = nn.LayerNorm(self.out_features)
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(
            self.out_features, 
            self.lstm_features, 
            num_layers=self.num_layers, 
            batch_first=True, 
        )
        self.dense_lstm = nn.Linear(self.lstm_features, self.lstm_features)
        self.dence_out = nn.Linear(self.out_features, self.out_features)
        self.pooler = nn.Linear(self.out_features, VOCAB_SIZE)
        self.dence_feature = nn.Linear(self.out_features, self.bert_hsize)
        self.softmax = nn.Softmax(dim=2) # use only eval mode.
        
        self.attention = build_attention(
            h_size=self.out_features,
            enc_size = self.lstm_features,
            dec_size = self.lstm_features,
            device=self.device
        )
        ###################################################

    def forward(self, key_value, h:torch.tensor, c:torch.tensor, response, mode='train'):
        """

        Args:
            key_value (torch.tensor): Encoder output vector sequence
            h (torch.tensor): Encoder LSTM output vector
            c (torch.tensor): Encoder LSTM Cell vector
            response (list): response word ID sequence
            mode (str, optional): 'train' or 'eval'. Defaults to 'train'.

        Returns:
            (outputs, (hn, cn), out_fvs)
            outputs (torch.tensor): Model output for CrossEntropyLoss
            hn (torch.tensor): Encoder LSTM output vector
            cn (torch.tensor): Encoder LSTM Cell vector
            out_fvs (torch.tensor): Model output for KLDivLoss
        """

        packs = pack_sequence([torch.tensor(t, device=self.device) for t in response], enforce_sorted=False)
        (model_input, lengths_info) = pad_packed_sequence(
            packs, 
            batch_first=True, 
            padding_value=0.0
        )
        
        if response == None:
            self.batch_size = 1
            self.seq_len = self.max_len
        else:
            self.batch_size = len(model_input)
            self.seq_len = len(model_input[0])
        h = h.clone()
        c = c.clone()

        if mode == 'train':
            result = self.train_forward(key_value, h, c, model_input)
        elif mode == 'eval':
            result = self.eval_forward(key_value, h, c)
        else:
            raise Exception('Invalid Mode :',mode)

        return result

    def train_forward(self, key_value, h, c, response:torch.tensor):
        inputs = response

        # Embedding & Dropout & Dense ##################### 
        embed = self.embedding(inputs.clone())
        affin = self.dense1(embed)
        affin = self.layer_norm0(affin)
        drop = self.dropout(affin)
        ###################################################

        # LSTM with LayerNorm #############################
        outs, (hn, cn) = self.lstm(drop, (h, c))

        # Attention #########################
        cntxt = self.attention(key_value, outs)
        ###################################################

        # Pooling Layer ###################################
        cntxt = self.dence_out(cntxt)
        out_fvs = self.dence_feature(cntxt)
        pooler_outputs = self.pooler(cntxt)
        ###################################################

        return pooler_outputs, (hn, cn), out_fvs

    def eval_forward(self, key_value, h, c):
        hn = h
        cn = c

        count = 0
        outputs=torch.zeros([self.batch_size, 1, VOCAB_SIZE]).to(self.device)
        out_fvs=torch.zeros([self.batch_size, 1, self.bert_hsize]).to(self.device)

        word = '[CLS]'
        inputs = [self.bert_tokenizer.convert_tokens_to_ids(word)] * self.batch_size

        while count < self.max_len:
            # Tokenizer ###################################
            input_tensor = torch.tensor(inputs, device=self.device)
            input_tensor = input_tensor.view([self.batch_size, 1])
            ###############################################

            # Embedding & Dropout & Dense #################
            embed = self.embedding(input_tensor.clone())
            affin = self.dense1(embed)
            affin = self.layer_norm0(affin)
            drop = self.dropout(affin)
            ###############################################

            # LSTM with LayerNorm #########################
            out_lstm, (hn, cn) = self.lstm(drop, (hn, cn))

            # Attention #########################
            cntxt = self.attention(key_value, out_lstm)
            ###############################################

            # Pooling Layer ###############################
            cntxt = self.dence_out(cntxt)
            dense_fvs = self.dence_feature(cntxt)
            pooler_outs = self.pooler(cntxt)
            ###############################################

            # to token ####################################
            batch_words = pooler_outs.view([self.batch_size, VOCAB_SIZE])
            (_, batch_ids) = batch_words.max(1)
            ###############################################

            inputs = batch_ids.tolist()
            outputs = torch.cat((outputs, pooler_outs), dim=1)
            out_fvs = torch.cat((out_fvs, dense_fvs), dim=1)
            count = count + 1

        return outputs[:,1:,:], (hn, cn), out_fvs[:,1:,:]
