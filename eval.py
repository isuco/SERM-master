"""
Run evaluation with saved models.
"""
import random
import argparse
import os
from tqdm import tqdm
import itertools
import torch
import numpy as np
from data.loader import DataLoader,get_long_tensor
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import json
from transformers.tokenization_albert import AlbertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,default="saved_models/03",help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test_rev_coref', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--cpu', action='store_true')

parser.add_argument("--max_seq_length", default=128, type=str)
parser.add_argument("--model_name_or_path", default="pretrained_models/albert_large/", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--vocab_file", default='spiece.model', type=str)
parser.add_argument("--spm_model_file", default="30k-clean.model", type=str)
parser.add_argument("--task_name", default=None, type=str,
                        help="The name of the task to train selected in the list: " )
parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
parser.add_argument("--config_name", default="config.json", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")

args = parser.parse_args()
print(torch.cuda.device_count())
args.cuda=True
# with open('err.json','r') as f:
#     data=json.load(f)
args.cpu=False

def addSpecialTokens(tokenizer):
    special_key = "additional_special_tokens"
    masks = []
    subj_entities = constant.SUBJ_ENTITIES
    obj_entities = constant.OBJ_ENTITIES
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    entities = constant.ENTITIES
    masks += ["ENTITY_" + e for e in entities]
    print(masks)
    masks += ['[EOE]']
    masks += ['[ESB]']
    masks += ['[EOB]']
    tokenizer.add_special_tokens({special_key: masks})

# def get_all_rank(l):
#     return itertools.permutations(l,len(l))

# ranks=get_all_rank(constant.ENTITIES)
# maxf1=-1
fixed_char={}
maxrank=constant.OBJ_ENTITIES
ranks=constant.OBJ_ENTITIES
entlen=len(ranks)
# while(len(fixed_char)<entlen):
#     for j,ner in enumerate(ranks):
#         maxf1=0
#         limitner=ner
#         for i,k in enumerate(ranks):
#             if i not in fixed_char.values():
#                 entities=[]
#                 for z in range(entlen):
#                     entities.append('[NULL'+str(z)+']')
#                 entities[i]=limitner
#                 constant.OBJ_ENTITIES=entities
#                 print(constant.ENTITIES)
tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path+"spiece.model")
    #model = AlbertModel.from_pretrained(args.model_name_or_path+"pytorch_model.bin",config=args.model_name_or_path+"config.json")
addSpecialTokens(tokenizer)
# opt['vocab_size'] = len(tokenizer)
# opt['albert_path'] = args.model_name_or_path + "pytorch_model.bin"
# opt['config_path'] = args.model_name_or_path + "config.json"

torch.manual_seed(args.seed)
random.seed(1024)
# if args.cpu:
#     args.cuda = False
# elif args.cuda:
torch.cuda.manual_seed(args.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"
print(torch.cuda.device_count())
# load opt
args.model_dir='saved_models/03'
#model_file=args.model_dir+'/best_model_aug.pt'
# print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
# trainer = GCNTrainer(opt)
# trainer.load(model_file)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
entitie_pairs = constant.REL_ENTITIES
predictions = []
key=[]
# for entitie_pair in entitie_pairs:
#     subj=entitie_pair[0]
#     obj=entitie_pair[-1]
with torch.cuda.device(0):
    #load vocab
    print(torch.cuda.current_device())
    # vocab_file = args.model_dir + '/vocab.pkl'
    # vocab = Vocab(vocab_file, load=True)
    # labels = constant.LABEL_TO_ID.keys()
    # lbstokens = []
    # for lbs in labels:
    #     lb = []
    #     if lbs == 'no_relation':
    #         subj = ['<UNK>']
    #         rels = '<UNK>'
    #     else:
    #         subj, rel = lbs.split(":")
    #         subj = (['SUBJ-PERSON'] if subj == 'per' else ['SUBJ-ORGANIZATION'])
    #         rels = rel.split("_")
    #     lb += map_to_ids(subj, vocab.word2id)
    #     lb += map_to_ids(rels, vocab.word2id)
    #     lbstokens.append(lb)
    # lbstokens = get_long_tensor(lbstokens, len(lbstokens),128)
    # subj="PERSON"
    # obj="NUMBER"
#     #
#     # assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
#     #
#     # # load data
#
#     # print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
#     #batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True,corefresolve=False)
#     #train_batch=DataLoader(data_file,opt['batch_size'],opt,vocab,evaluation=True,corefresolve=False,is_aug=True)
#     # train_aug_batch=DataLoader(data_file,opt['batch_size'],opt,vocab,evaluation=True,corefresolve=False)
#     # helper.print_config(opt)

#     SUBJ_LIST=list(constant.SUBJ_NER_TO_ID.keys())
#     OBJ_LIST=list(constant.OBJ_NER_TO_ID.keys())




    #train_batch.setisEval(True)
    # probs={}
    # for i,batch in enumerate(train_batch):
    #     preds, prob, _,sample = trainer.predict(batch)
    #     probs=dict(probs,**prob)
    # # probs = sorted(probs.items(),key=lambda item:max(item[1]))
    # train_batch.LabeledAugData(probs)
    # train_batch.setisEval(False)
    # for i, b in enumerate(batch_iter):
    #     preds, prob, _,sample = trainer.predict(b)
    #     predictions += preds
    #     all_probs += prob
    #     samples = dict(samples, **sample)
    # SUBJ_LIST=list(constant.SUBJ_NER_TO_ID.keys())
    # OBJ_LIST=list(constant.OBJ_NER_TO_ID.keys())
    # SUBJ_LIST.remove('<PAD>')
    # SUBJ_LIST.remove('<UNK>')
    # OBJ_LIST.remove('<PAD>')
    # OBJ_LIST.remove('<UNK>')

    # print(torch.cuda.device_count())
    # with torch.cuda.device(1):
    #     for subj in SUBJ_LIST:
    #         for obj in OBJ_LIST:
    # print("eval samples of subj:"+subj+" obj:"+obj)
    # args.model_dir = 'saved_models/02'
    # if os.path.exists(args.model_dir+'/'+subj+"_"+obj+"_"+"best_model.pt"):
    #     model_file = args.model_dir +'/'+subj+"_"+obj+"_"+"best_model.pt"
    # else:
    #     model_file = args.model_dir + '/best_model.pt'
    model_file=args.model_dir+'/best_model_809' \
                              '.pt'
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
    # trainer = GCNTrainer(opt,lbstokens=lbstokens)


    trainer = GCNTrainer(opt)

    trainer.load(model_file)
    batch = DataLoader([data_file], 32, opt, tokenizer, evaluation=True, corefresolve=True)
    batch_iter = tqdm(batch)

    all_probs = []
    samples = []
    for bi, b in enumerate(batch_iter):
        preds, probs, _,sample= trainer.predict(b)
        predictions += preds
        all_probs += probs
        # effsum+=lab_eff
        # lab_nums+=lab_num
        samples=samples+sample

    key+=batch.gold()

    with open('samples.json','w') as f:
        json.dump(samples,f,indent=4)

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch, predictions, verbose=True)
# if f1>maxf1:
#     maxf1=f1
#     fixed_char[ner]=i
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
    # print(fixed_char)
    # with open('entities.json', 'w') as f:
    #     json.dump(fixed_char, f, indent=4)

print("Evaluation ended.")

