import math

import torch
import torch.nn as nn

def build_attention(h_size, enc_size, dec_size, device):
    attention_layer = AttentionLayer(
        h_size=h_size,
        enc_size=enc_size,
        dec_size=dec_size,
        device=device
    )
    return attention_layer

class AttentionLayer(nn.Module):
    def __init__(self, h_size, enc_size, dec_size, device='cuda:0'):
        super(AttentionLayer, self).__init__()

        # Parameters ######################################
        self.h_size = h_size
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.scale_factor = math.sqrt(self.h_size)
        self.device = device
        ###################################################

        # Layers ##########################################
        self.query_dense = nn.Linear(self.h_size, self.h_size)
        self.key_dense = nn.Linear(self.dec_size, self.h_size)
        self.value_dense = nn.Linear(self.dec_size, self.h_size)

        self.softmax = nn.Softmax(dim=2)

        self.output_dense = nn.Linear(self.h_size, self.h_size)
        self.shape_dence = nn.Linear(self.h_size + self.dec_size, self.h_size)

        self.layer_norm = nn.LayerNorm(self.h_size)
        ###################################################

    def forward(self, encoder_out:torch.tensor, decoder_out:torch.tensor):
        """
        Attributes
        ----------
        encoder_out : 
            type : tensor(torch.float32)
            shape : [batch_size, seq_len, h_size]
            This is encoder outputs.
            Its roll in Attention is Key and Value.
        decoder_out :
            type : tensor(torch.float32)
            shape : [N, batch_size, h_size]
            This is decoder outputs.
            Its roll in Attention is Query.

        Returns
        -------
        final_output : 
            type : tensor(torch.float32)
            shape : [N, batch_size, h_size]
            Context vector + decoder_output.
        """

        key = encoder_out.clone().to(device=self.device)
        value = encoder_out.clone().to(device=self.device)
        query = decoder_out.clone().to(device=self.device)

        # Dense Layers ####################################
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.query_dense(query)

        query = query/self.scale_factor
        ###################################################

        # Mat Mul Layer ###################################
        logit = torch.bmm(query, key.transpose(1,2))

        attention_weight = self.softmax(logit)

        output = torch.bmm(attention_weight, value)
        context_vec = self.output_dense(output)
        ###################################################

        # Add Layer #######################################
        final_out = torch.cat((decoder_out, context_vec), axis=-1)
        final_out = self.shape_dence(final_out)
        final_out = self.layer_norm(final_out)
        ###################################################

        return final_out