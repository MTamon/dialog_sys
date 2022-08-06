from tqdm import tqdm
from argparse import ArgumentParser
from logger_gen import set_logger

from nn.encoder_decoder import build_model
from utils.dataloader import textDataset, idDataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM


def add_args(parser: ArgumentParser):
    parser.add_argument('--model-path', default=None, type=str, help='training target model (.pth)')
    parser.add_argument('--epoch-num', default=1, type=int, help='number of epoch')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size of train')
    parser.add_argument('--enc-len', default=128, type=int, help='intput sequence length of encoder')
    parser.add_argument('--max-len', default=62, type=int, help='generated sequence length')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='one of beta value of AdamW')
    parser.add_argument('--beta2', default=0.98, type=float, help='one of beta value of AdamW')
    parser.add_argument('--weight-decay', default=1e-2, type=float, help='weight decay coefficient of AdamW')
    parser.add_argument('--criterion-reduction', default='mean', type=str, help='reduction which is parameter of criterion')
    parser.add_argument('--adamw-eps', default='1e-8', type=float, help='eps which is parameter of optimizer AdamW')
    parser.add_argument('--disp-progress', default=False, action='store_true', help='display training progress')
    parser.add_argument('--train-data', default='data/train', type=str, help='path of training data')
    parser.add_argument('--eval-data', default='data/eval', type=str, help='path of evaluation data')
    parser.add_argument('--bert-path', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/pytorch_model.bin', type=str, help='path of BERT model')
    parser.add_argument('--bert-config', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/bert_config.json', type=str, help='path of BERT config file')
    parser.add_argument('--bert-vocab', default='resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/vocab.txt', type=str, help='path of BERT torkenizer vocab')
    
    parser.add_argument('--cross-loss-coef', default=0.5, type=float, help='ocoefficient of CrossEntropyLoss')
    parser.add_argument('--mse-loss-coef', default=0.5, type=float, help='coefficient of MSELoss')
    
    parser.add_argument('--num-layer', default=2, type=int, help='LSTM layer number')
    parser.add_argument('--use-bidirectional', default=False, action='store_true', help='use BiDirectioal LSTM')
    parser.add_argument('--bert-hsize', default=768, type=int, help='hidden vector size of BERT')
    parser.add_argument('--hidden-size', default=256, type=int, help='hidden layer size')
    parser.add_argument('--lstm-hidden-size', default=256, type=int, help='hidden layer size')
    
    return parser

class TrainModel(object):
    def __init__(self, args, logger, device='cpu'):
        self.args = args
        self.logger = logger
        self.device = device
        
        # load BERT model
        self.bert_config = BertConfig.from_json_file(self.args.bert_config)
        self.bert_model = BertModel.from_pretrained(self.args.bert_path, config=self.bert_config)
        self.bert_mask = BertForMaskedLM.from_pretrained(self.args.bert_path, config=self.bert_config)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.bert_vocab, do_lower_case=False, do_basic_tokenize=False)
        self.vocab_size = self.bert_tokenizer.vocab_size
        
        self.bert_mask = self.bert_mask.to(device=self.device)
        
        # make dataloader
        dataloader0 = idDataLoader(self.args.train_data, self.args)
        dataset0 = textDataset(dataloader0, batch_size=self.args.batch_size)
        dataloader1 = idDataLoader(self.args.eval_data, self.args)
        dataset1 = textDataset(dataloader1, batch_size=self.args.batch_size)
        
        dataloader_train = dataset0
        dataloader_test = dataset1

        self.dataloader = {'train':dataloader_train, 'eval':dataloader_test}
        
        # building model
        self.model = build_model(
            self.args,
            self.bert_model,
            self.bert_tokenizer,
            self.device
        )
        
        # setting criterion & optimizer
        self.criterion1 = nn.CrossEntropyLoss(
            ignore_index=0
        )
        #self.criterion2 = nn.MSELoss(
        self.criterion2 = nn.KLDivLoss(
            reduction=self.args.criterion_reduction
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.adamw_eps,
            weight_decay=self.args.weight_decay
        )
        
        # load model or make new model param file
        try:
            net_dic = torch.load(args.model_path, map_location=device)
            logger.info('loading model ... ')
            self.model.load_state_dict(net_dic)
            logger.info('Done.')
        except Exception:
            logger.info('{} does not exist.'.format(args.model_path))
            logger.info('save model to {}'.format(args.model_path))
            torch.save(self.model.state_dict(), args.model_path)
            
        self.model.to(device=self.device)
        
        # acceleration when forward propagation and loss function calculation methods are constant.
        torch.backends.cudnn.benchmark = True
        # accelerate GPU & save memory
        self.scaler = torch.cuda.amp.GradScaler()

class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        self.epoch_num = self.args.epoch_num
        self.max_len = self.args.max_len
        self.batch_size = self.args.batch_size

    def __call__(self, tm: TrainModel):

        train_batch_loss = []
        eval_batch_loss = []
        train_batch_acc = []
        eval_batch_acc = []

        for epoch in range(self.epoch_num):
            self.logger.info('Epoch {}/{}'.format(epoch+1, self.epoch_num))

            for phase in ["train", "eval"]:
                if phase == "train":
                    tm.model.train()
                else:
                    tm.model.eval()

                score = self.calc_loss(tm, phase)

                if phase == "train":
                    train_batch_loss.append(score['loss'])
                    train_batch_acc.append(score['acc'])
                else:
                    eval_batch_loss.append(score['loss'])
                    eval_batch_acc.append(score['acc'])
                
        return (train_batch_loss, eval_batch_loss, train_batch_acc, eval_batch_acc)
                
    def calc_loss(self, tm:TrainModel, phase):
        epoch_loss = 0.0
        epoch_corrects = 0
        epoch_data_sum = 0

        turn = 0
        for query, response in tqdm(tm.dataloader[phase]):

            # make teacher data
            packs = pack_sequence([torch.tensor(t, device=device).clone().detach() for t in response], enforce_sorted=False)
            (answer, _) = pad_packed_sequence(
                packs, 
                batch_first=True, 
                padding_value=0.0
            )

            if len(answer[0]) < self.max_len:
                zero_pad = torch.zeros([self.batch_size, (self.max_len - len(answer[0]))], dtype=int, device=device)
                answer = torch.cat((answer, zero_pad), dim=1).contiguous()
            if len(answer[0]) > self.max_len:
                answer = answer[:,:self.max_len].contiguous()

            answer_resp = answer[:,1:].clone().detach() # remove [CLS]
            answer_resp = torch.cat((answer_resp, torch.zeros([self.batch_size, 1], dtype=int, device=device)), dim=1)
            answer_resp = answer_resp.contiguous()

            input_decoder = answer
            input_decoder = input_decoder.tolist()

            answer_mask = torch.ones(answer_resp.shape, dtype=int, device=tm.device) * (answer_resp != 0)
            # learning [PAD]
            answer_mask = answer_mask

            # initialize
            tm.optimizer.zero_grad()

            # Forward propagation
            with torch.set_grad_enabled(phase=="train"), torch.cuda.amp.autocast():
                outputs, feature_vecs = tm.model([query, input_decoder], phase)
                outputs.contiguous()
                
                # for loss1
                vocab_size = tm.vocab_size
                outputs_cri = outputs.contiguous().view([-1, vocab_size]) # [batch_size, seq_len, vocab_size] >> [(batch_size * seq_len), vocab_size]
                outputs_cri = outputs_cri.contiguous()
                answer_resp_cri = answer_resp.view(-1) # [batch_size, seq_len] >> [(batch_size * seq_len)]
                
                # for loss2
                outputs_lsm = F.log_softmax(outputs, dim=-1)
                answer_mask = torch.ones(answer_resp.shape ,dtype=torch.int32, device=device) * (answer_resp != 0)
                pre_answer = tm.bert_mask(answer_resp, attention_mask=answer_mask)
                pre_answer = F.log_softmax(pre_answer[0], dim=-1)
                
                # Loss for cross entropy
                loss1 = tm.criterion1(outputs_cri, answer_resp_cri)
                loss2 = tm.criterion2(outputs_lsm, pre_answer)
                loss = args.cross_loss_coef * loss1 + args.mse_loss_coef * loss2

                # Prediction
                _, preds = torch.max(outputs, 2)
                preds = preds * answer_mask

                # Learning
                if phase=="train":
                    tm.scaler.scale(loss).backward()
                    tm.scaler.step(tm.optimizer)
                    tm.scaler.update()

                # Loss is mean of batch. So total batch loss is bellow.
                epoch_loss += loss.item() * self.batch_size
                # correct
                epoch_corrects += torch.sum(preds == answer_resp) - torch.sum(answer_mask==0)
                epoch_data_sum += torch.sum(answer_mask)
            
            turn += 1

        torch.save(tm.model.state_dict(), self.args.model_path)
        logger.info('complete saving model')

        # display loss & acc for each epoch
        epoch_loss = epoch_loss / len(tm.dataloader[phase])
        epoch_acc = epoch_corrects.double() / epoch_data_sum

        logger.info("{} Loss {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
        score = {}
        score['loss'] = epoch_loss
        score['acc'] = epoch_acc
        
        return score


# setup argument parser & logger
parser = ArgumentParser('This program is trainer for seq2seq with attention model.')
parser = add_args(parser)
args = parser.parse_args()
logger = set_logger("training", "log/train.log")

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Device: {}".format(device))

trainset = TrainModel(args, logger, device)
trainer = Trainer(args, logger)

# training model
loss = trainer(trainset)