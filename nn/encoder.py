from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn

from transformers import BertModel

def build_encoder(num_layers, bert_model, out_features, lstm_features, bidrectional, enc_len, device='cuda:0'):
    """Generate encoder model.

    Args:
        num_layers (int): LSTM layer number.
        bert_model (BertModel): used Bert model.
        out_features (int): encoder output feature size.
        lstm_features (int): lstm hidden size.
        bidrectional (bool): use bidirectional lstm when bidirectional=True
        enc_len (int): max encoder sequence length.
        device (str): device type. Defaults to 'cuda:0'.

    Returns:
        Encoder(nn.Module): encoder module
    """
    model = Encoder(
        bert_model=bert_model,
        out_features=out_features,
        lstm_hidden_size=lstm_features,
        num_layers=num_layers,
        bidirectional=bidrectional,
        enc_len=enc_len,
        device=device
    )

    return model

class BertEmbedding(nn.Module):
    """
    same nn.Embedding(num_input=32006, emb_size=out_features, padding_idx=0)
    """
    def __init__(self, bert_model=None, out_features=256, device='cuda:0'):
        super(BertEmbedding, self).__init__()

        # Parameters ######################################
        self.out_features = out_features
        self.device = device
        ###################################################

        # Layers ##########################################
        self.bert = bert_model
        self.freez(self.bert) # freeze auto grad
        self.transfer_learning(self.bert)

        self.pooler_dense = nn.Linear(768, self.out_features)
        self.layer_norm = nn.LayerNorm(self.out_features)
        ###################################################

    def forward(self, inputs:torch.tensor):
        """
        Bert Embedding or Bert Encoder Module.

        Attribute
        ---------
        input : 
            type : torch.tensor(int)
            shape : [batch_size, seq_len]
            Input sequence list which is refilled [PAD].
            You need to analyse with proper morphological analyzer for which you use bert model.

        Return
        ------
        output :
            type : torch.tensor(float)
            shape : [batch_size, seq_len, out_features]
            Embedding expression by BERT.
            Padding indices is initialized by 0.0.
        """

        # Generate mask ###################################
        self.source_mask = torch.ones(inputs.shape, dtype=int, device=self.device) * (inputs != 0)
        ###################################################

        # Layers : BERT -> Linear -> LayerNorm ############
        output, _ = self.bert(inputs, self.source_mask)
        output = self.pooler_dense(output)
        output = self.layer_norm(output)
        ###################################################

        # Embedding mask to padding index #################
        self.source_mask = self.source_mask.to(torch.float32)
        output = output * self.source_mask.unsqueeze(-1) 
        ###################################################

        return output

    def freez(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def unfreez(self, module):
        for p in module.parameters():
            p.requires_grad = True
            
    def transfer_learning(self, module:BertModel):
        module.pooler.dense = nn.Linear(768, 768, bias=True)
        module.pooler.dense.requires_grad = True

class Encoder(nn.Module):
    def __init__(self, bert_model=None, out_features=256, lstm_hidden_size=256, num_layers=1, bidirectional=False, enc_len=128, device='cuda:0'):
        """initialize encoder LSTM

        Args:
            out_features (int): output feature size. Defaults to 256.
            lstm_hidden_size (int): lstm hidden vector size. Defaults to 256.
            num_layers (int): lstm layer number. Defaults to 1.
            bidirectional (bool): Defaults to False
            enc_len (int): encoder's sequence length. Defaults to 128.
            device (str): device type. Defaults to 'cuda:0'.
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.out_features = out_features
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.enc_len = enc_len
        self.device = device

        self.bert_embed = BertEmbedding(
            bert_model=bert_model,
            out_features=self.out_features,
            device=self.device
        )
        self.embedding = nn.Embedding(32006, 768, padding_idx=0)
        self.input_dense = nn.Linear(768, self.out_features)
        self.input_norm = nn.LayerNorm(self.out_features)
        self.input_drop = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(
            input_size=self.out_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )
        
        directions = 2 if self.bidirectional else 1
        self.directions = directions
        
        d_h = [nn.Linear(self.lstm_hidden_size*directions, self.lstm_hidden_size).to(device) for i in range(num_layers)]
        d_c = [nn.Linear(self.lstm_hidden_size*directions, self.lstm_hidden_size).to(device) for i in range(num_layers)]
        
        self.dense_h = nn.ModuleList(d_h)
        self.dense_c = nn.ModuleList(d_c)
        
        self.dense = nn.Linear(self.out_features, self.out_features)
        
        normh = [nn.LayerNorm(self.lstm_hidden_size).to(device) for i in range(num_layers)]
        normc = [nn.LayerNorm(self.lstm_hidden_size).to(device) for i in range(num_layers)]
        
        self.layer_normh = nn.ModuleList(normh)
        self.layer_normc = nn.ModuleList(normc)
        
        self.layer_norm = nn.LayerNorm(self.out_features)

    def forward(self, ids:list):
        """
        Attributes
        ----------
        ids : 
            type : list
            shape : [batch_size, seq_len]
            Sentence whose token changed ids.
        """
        
        batch_size = len(ids)

        ids = [seq[-self.enc_len:] for seq in ids]
        packs = pack_sequence([torch.tensor(t, device=self.device) for t in ids], enforce_sorted=False)
        (model_input, lengths_info) = pad_packed_sequence(
            packs, 
            batch_first=True, 
            padding_value=0.0
        )
        

        # BERT Embedding Layer ############################
        outputs = self.bert_embed(model_input)
        ###################################################

        # Generate mask ###################################
        dense_mask = self.bert_embed.source_mask.detach()
        dense_mask = dense_mask.unsqueeze(-1)
        ###################################################

        # Embedding for LSTM ##############################
        lstm_input = self.embedding(model_input)
        lstm_input = self.input_dense(lstm_input)
        lstm_input = self.input_norm(lstm_input)
        lstm_input = self.input_drop(lstm_input)
        ###################################################

        # Don't Need mask ! (hint : length_info)
        lstm_packs = pack_padded_sequence(
            lstm_input,
            lengths=lengths_info, 
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM Layer ####################################
        _, (hn, cn) = self.lstm(lstm_packs)
        
        hn = hn.view([self.directions, self.num_layers, batch_size, self.lstm_hidden_size])
        hn = torch.transpose(hn, 1, 2)
        hn = torch.transpose(hn, 0, 2)
        hn = hn.reshape([self.num_layers, batch_size, self.directions*self.lstm_hidden_size])
        
        cn = cn.view([self.directions, self.num_layers, batch_size, self.lstm_hidden_size])
        cn = torch.transpose(cn, 1, 2)
        cn = torch.transpose(cn, 0, 2)
        cn = cn.reshape([self.num_layers, batch_size, self.directions*self.lstm_hidden_size])
        
        batch_size = len(outputs)
        ###################################################

        # Pooler ##############################

        outputs = self.dense(outputs) # one Attention use.
        outputs = self.layer_norm(outputs)
        outputs = outputs*dense_mask

        hn = [dense(h) for dense, h in zip(self.dense_h, hn)]
        hn = [norm(h) for norm, h in zip(self.layer_normh, hn)]
        hn = torch.cat([h.unsqueeze(0) for h in hn], dim=0)

        cn = [dense(c) for dense, c in zip(self.dense_c, cn)]
        cn = [norm(c) for norm, c in zip(self.layer_normc, cn)]
        cn = torch.cat([c.unsqueeze(0) for c in cn], dim=0)

        return (outputs, hn, cn)