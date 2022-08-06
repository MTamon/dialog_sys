from argparse import ArgumentParser

import torch
from transformers import BertTokenizer, BertModel, BertConfig

from nn.encoder_decoder import build_model
from ngram.scoring.score_sentence import ScoreSentence
from logger_gen import set_logger
from BJtokenizer import BertWithJumanModel, remove_duplication


def add_args(parser: ArgumentParser):
    parser.add_argument('--model-path', default=None, type=str, help='training target model (.pth)')
    parser.add_argument('--bert-path', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/pytorch_model.bin', type=str, help='path of BERT model')
    parser.add_argument('--bert-config', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/bert_config.json', type=str, help='path of BERT config file')
    parser.add_argument('--bert-vocab', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/vocab.txt', type=str, help='path of BERT torkenizer vocab')
    parser.add_argument('--enc-len', default=128, type=int, help='intput sequence length of encoder')
    parser.add_argument('--max-len', default=62, type=int, help='generated sequence length')
    
    parser.add_argument('--num-layer', default=2, type=int, help='LSTM layer number')
    parser.add_argument('--use-bidirectional', default=False, action='store_true', help='use BiDirectioal LSTM')
    parser.add_argument('--bert-hsize', default=768, type=int, help='hidden vector size of BERT')
    parser.add_argument('--hidden-size', default=256, type=int, help='hidden layer size')
    parser.add_argument('--lstm-hidden-size', default=256, type=int, help='hidden layer size')
    
    parser.add_argument('--used-ngram-model', default='scoring/models/bccwj-csj-np.bin', type=str, help='n-gram model for KenLM scoring')
    parser.add_argument('--remove-limit', default=-3.0, type=float, help='cut model\'s response by n-gram scoring')
    parser.add_argument('--resp-gen', default=30, type=int, help='number of times to generate response sentences')
    
    parser.add_argument('--length-dist-mean', default=10, type=int, help='mean of length that generated sentence')
    parser.add_argument('--length-dist-var', default=5, type=int, help='variance of length that generated sentence')
    parser.add_argument('--length-dist-scale', default=1, type=int, help='variance of length that generated sentence')
    
    return parser

# setup argument parser & logger
parser = ArgumentParser('This program is dialogue sysytem.')
parser = add_args(parser)
args = parser.parse_args()
logger = set_logger("dialog", "log/dialog.log")

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Device : {}".format(device))

# load BERT model
bert_config = BertConfig.from_json_file(args.bert_config)
bert_model = BertModel.from_pretrained(args.bert_path, config=bert_config)
bert_tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=False, do_basic_tokenize=False)
vocab_size = bert_tokenizer.vocab_size

model = build_model(
    args=args,
    bert_model=bert_model,
    bert_tokenizer=bert_tokenizer,
    device=device
)
model.to(device=device)

# tokenizer
bert_juman = BertWithJumanModel(args, logger)
# N-gram scoring
score = ScoreSentence(args=args)

# load model & set parameters
try:
    net_dic = torch.load(args.model_path, map_location=device)
    logger.info('loading model ... ')
    model.load_state_dict(net_dic)
    logger.info('Done.')
except Exception as e:
    logger.info('{} does not exist.'.format(args.model_path))

# Use as is train mode to generate various responses
model.train()

# processing of dialog
while True:
    input_s = input('>>')
    (ids, recv) = bert_juman.get_sentence_ids(input_s)
    logger.info(ids)

    logger.info(ids)
    logger.info(recv)
    
    # create model input
    querys = [ids] * args.resp_gen
    answrs = [[0]] * args.resp_gen
    model_inputs = [querys, answrs]
    
    # inference
    outs, _ = model(model_inputs, "eval")
    outs = outs.contiguous()
    _, preds = torch.max(outs, 2)

    responses = []
    res_tokens = []
    for pred in preds:
        
        # get string of response
        resp = bert_juman.get_sentence(pred.tolist())
        res_tokens.append(resp)
        
        # get tokens of response
        tokens = bert_juman.convert_ids_to_tokens(pred)
        
        # remove duplication & scoring
        sentence = remove_duplication(tokens)
        sentence = bert_juman.div_sep(sentence)[0]
        scores, sentence = score(' '.join(sentence))
        logger.info('cand: {}, {}'.format(sentence, scores))
        
        responses.append([sentence, scores])
        
    # choice best score
    max_score = -100000.0
    max_ind = 0
    best_sentence = []
    for i, response in enumerate(responses):
        sentence, sent_score = response
        if sent_score > max_score:
            max_score = sent_score
            max_ind = i
            best_sentence = sentence
    logger.info('Before (duplication): {}'.format(res_tokens[max_ind])) # Before removing duplication
    logger.info('After (duplication): {}'.format(responses[max_ind])) # After removing duplication
    logger.info('sys: {}'.format(bert_juman.tokens_to_sentence(best_sentence)))