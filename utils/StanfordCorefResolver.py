import json
from stanfordcorenlp import StanfordCoreNLP
from functools import reduce
import concurrent.futures as fu
import copy

nlp = StanfordCoreNLP("../../StanfordNLP/stanford-corenlp-latest/stanford-corenlp-4.1.0")

def trimstring(s):
    s=s.replace("''","(")
    s=s.replace("'","")
    s=s.replace("(","'")
    s=s.replace("``","`")
    s=s.replace("%40","@")
    s=s.replace("%2B","+")
    return s


def processchar(s):
    if "`" in s:
        s="`"
    return s

def fixchar(s):
    s=s.replace(']','')
    return s

def trimspace(s):
    s=s.replace(' ','')
    return s

def indexOfSpan(span,tokens):
    tmptokens=copy.deepcopy(tokens)
    tmpspan=copy.deepcopy(span)
    tmptokens=list(map(trimstring,tmptokens))
    tmpspan = list(map(trimstring, tmpspan))
    index=-1
    isfind=False
    for j in range(len(tmptokens)):
        isfind=True
        for i in range(len(tmpspan)):
            if tmpspan[i]!=tmptokens[i+j]:
                if '/' in tmptokens[i+j]:
                    i = tmptokens[i+j].index('/')
                    if tmpspan[i]!=tmptokens[i+j][:i] and tmpspan[i]!=tmptokens[i+j][(i+1):]:
                        isfind=False
                        break
        if isfind:
            index=j
            break
    assert index!=-1
    return index

def getCorefSpan(span,tokens):
    coref_list=[]
    span_len=len(span)
    span = " ".join(span)
    if span.islower() and span_len<=1:
        return coref_list
    else:
        for idx in range(len(tokens)-1):
            t_span=tokens[idx:idx+span_len]
            t_span=" ".join(t_span)
            if abs(len(t_span)-len(span))<3 and span in t_span:
                coref_list.append(list(range(idx,idx+span_len)))
        return coref_list

def generateCorefList(d):
    tokens = list(d['token'])
    tokens = list(map(fixchar, tokens))
    raw_tokens = copy.deepcopy(tokens)
    tokens = list(map(processchar, tokens))
    tokens = list(map(trimstring, tokens))
    # index = 0
    # sent = " ".join(tokens)
    # max_id = len(tokens) - 1
    # doc = nlp.word_tokenize(sent)
    # doc=list(map(trimspace,doc))
    # i = 0
    # j = 0
    # biasmap = {}
    # ismatch = True
    # while (i < len(tokens) and j < len(doc)):
    #     if tokens[i].strip()=="":
    #         i=i+1
    #     elif tokens[i].strip()==doc[j].strip():
    #         biasmap[j]=i
    #         i=i+1
    #         j=j+1
    #     else:
    #         ismatch = False
    #         if doc[j] in tokens[i]:
    #             docspan = doc[j]
    #             tokens[i]=tokens[i].strip()
    #             k = 0
    #             while (docspan in tokens[i]):
    #                 if ((doc[j + k + 1] == '-' or doc[j + k + 1] == "/") and len(doc)>j+k+2):
    #                     docspan = docspan + doc[j + k + 1] + doc[j + k + 2]
    #                     if docspan == tokens[i]:
    #                         for z in range(j, j + k + 3):
    #                             biasmap[z] = i
    #                         j = j + k + 3
    #                         i = i + 1
    #                         ismatch = True
    #                         break
    #                     k = k + 2
    #                 else:
    #                     docspan = docspan + doc[j + k + 1]
    #                     if docspan == tokens[i]:
    #                         for z in range(j, j + k + 2):
    #                             biasmap[z] = i
    #                         j = j + k + 2
    #                         i = i + 1
    #                         ismatch = True
    #                         break
    #                     k = k + 1
    #         elif tokens[i] in doc[j]:
    #             tokenspan = tokens[i]
    #             k=0
    #             while (tokenspan in doc[j]):
    #                 if ((tokens[i + k + 1] == '-' or tokens[i + k + 1] == "/" )and len(tokens)<i+k+2):
    #                     tokenspan = tokenspan + tokens[i + k + 1] + tokens[i + k + 2]
    #                     if tokenspan == doc[j]:
    #                         for z in range(i, i + k + 3):
    #                             biasmap[j] = z
    #                         i = i + k + 3
    #                         j = j + 1
    #                         ismatch = True
    #                         break
    #                     k = k + 2
    #                 else:
    #                     tokenspan = tokenspan + tokens[i + k + 1]
    #                     if tokenspan == doc[j]:
    #                         for z in range(i, i + k + 2):
    #                             biasmap[j] = z
    #                         i = i + k + 2
    #                         j = j + 1
    #                         ismatch = True
    #                         break
    #                     k = k + 1
    #     if not ismatch:
    #         break
    # if not ismatch:
    #     print(d['id'])
    #     return d
    # # try:
    # biasmap[len(doc)]=len(tokens)
    # clusters = nlp.coref(sent)

    ss, se = d['subj_start'], d['subj_end']
    subj_span = tokens[ss:se + 1]
    os, oe = d['obj_start'], d['obj_end']
    obj_span = tokens[os:oe + 1]
    # d['token']=processtokens
    # subj_list = [list(range(ss, se + 1))]
    # obj_list = [list(range(os, oe + 1))]
    raw_subj=list(range(ss, se + 1))
    raw_obj=list(range(os,oe+1))
    subj_list=getCorefSpan(subj_span,tokens)
    obj_list=getCorefSpan(obj_span,tokens)

    if raw_subj not in subj_list:
        subj_list.append(raw_subj)
    if raw_obj not in obj_list:
        obj_list.append(raw_obj)
    # subj_index = reduce(lambda x, y: x + y, subj_list)
    # obj_index = reduce(lambda x, y: x + y, obj_list)
    # subj_span = " ".join(subj_span)
    # for cluster in clusters:
    #     mentions = [mention[-1] for mention in cluster]
    #     if subj_span in mentions:
    #         for mention in cluster:
    #             biasss=biasmap[mention[1] - 1]
    #             biasse=biasmap[mention[2] - 2]
    #             if biasss != ss and biasse-biasss<5:
    #                 csspan = list(range(biasss,biasse+1))
    #                 if all([(i not in subj_index) for i in csspan]):
    #                     subj_list.append(csspan)
    #     if obj_span in mentions:
    #         for obj_span in mentions:
    #             biasos = biasmap[mention[1] - 1]
    #             biasoe = biasmap[mention[2] - 2]
    #             if biasos != os and biasoe-biasos<5:
    #                 cospan = list(range(biasos, biasoe+1))
    #                 if all([(i not in obj_index) for i in cospan]):
    #                     obj_list.append(cospan)

    d['subj_list'] = subj_list
    d['obj_list'] = obj_list
    d['tokens'] = raw_tokens
    # except:
    #     print(d['id'])
    return d

if __name__=='__main__':
    with open('../dataset/tacred/test.json') as infile:
        datas = json.load(infile)
    for d in datas:
        generateCorefList(d)
    with open('train_rev_coref.json','w') as f:
        json.dump(datas,f)