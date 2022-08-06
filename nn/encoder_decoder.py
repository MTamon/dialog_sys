import torch.nn as nn

from .encoder import build_encoder
from .decoder import build_decoder

def build_model(args, bert_model, bert_tokenizer, device):
    model = EncoderDecoder(args, bert_model, bert_tokenizer, device)
    return model

class EncoderDecoder(nn.Module):
    """
    Synthesize models
    """

    def __init__(self, args, bert_model, bert_tokenizer, device):
        super(EncoderDecoder, self).__init__()

        self.encoder = build_encoder(
            num_layers=args.num_layer,
            bert_model=bert_model,
            out_features=args.hidden_size,
            lstm_features=args.lstm_hidden_size,
            bidrectional=args.use_bidirectional,
            enc_len=args.enc_len,
            device=device
        )
        self.decoder = build_decoder(
            num_layers=args.num_layer,
            out_features=args.hidden_size,
            lstm_features=args.lstm_hidden_size,
            bert_tokenizer=bert_tokenizer,
            max_len=args.max_len,
            bert_hsize=args.bert_hsize,
            device=device
        )

    def forward(self, inputs, mode):
        """
        Parameters
        ----------
        inputs : 
            type : list
            shape : [2, batch_size, seq_len]
            [0,:,:] is query.
            [1,:,:] is response.
        mode : 
            type : string
            'train' or 'eval'
        """
        # Encoder #########################################
        hs, h, c = self.encoder(inputs[0])
        ###################################################

        # Decoder #########################################
        outputs, (_, _), out_fvs = self.decoder(
            hs, 
            h,
            c,
            response=inputs[1],
            mode=mode
        )
        ###################################################

        return outputs, out_fvs