"""
Basic operations on trees.
"""

import numpy as np
from utils import constant
import copy

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        # self.subjdis=constant.MAX_DIS
        # self.objdis=constant.MAX_DIS
        self.sdpdis=constant.MAX_DIS
        self.depnode=-1
        self.domainids=set()
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def getSDPAncestor(root,t1,t2):
    if root is None or (t1==root and t2==root):
        root.sdpdis=0
        return 0,root
    # if t1==root:
    #     root.sdpdis=0
    #     cur=t2
    #     while cur!=t1:
    #         cur.sdpdis=0
    #         cur=cur.parent
    #     return 0,root
    sub_path = [root]
    obj_path = [root]
    cur=t1
    while cur!= root:
        sub_path.insert(1, cur)
        cur=cur.parent
    cur =t2
    while cur!= root:
        obj_path.insert(1, cur)
        cur=cur.parent

    length = min(len(sub_path),len(obj_path))
    index = length-1
    for i in range(length):
        if sub_path[i]!=obj_path[i]:
            index=i-1
            break
    sdp_length=len(sub_path)+len(obj_path)-2*index-1
    for j in range(index,len(sub_path)):
        sub_path[j].sdpdis=0
    for j in range(index,len(obj_path)):
        obj_path[j].sdpdis=0
    return sub_path[index],sdp_length


def getSDPAncestorEval(root,pairlist):
    minsdplength=constant.INFINITY_NUMBER
    ancester_id=-1
    sub_path,obj_path=None,None
    select_pair=None
    for i in range(len(pairlist)):
        subj = pairlist[i][0]
        obj = pairlist[i][-1]

    # if t1==root:
    #     root.sdpdis=0
    #     cur=t2
    #     while cur!=t1:
    #         cur.sdpdis=0
    #         cur=cur.parent
    #     return 0,root
        s_path = [root]
        o_path = [root]
        cur = subj
        while cur != root:
            s_path.insert(1, cur)
            cur = cur.parent
        cur = obj
        while cur != root:
            o_path.insert(1, cur)
            cur = cur.parent

        length = min(len(s_path),len(o_path))
        index = length-1
        for k in range(length):
            if s_path[k] != o_path[k]:
                index = k - 1
                break
        sdp_length=len(s_path)+len(o_path)-2*index-1
        if sdp_length<minsdplength:
            obj_path=o_path
            sub_path=s_path
            ancester_id=index
            minsdplength=sdp_length
            select_pair=i
    ancester=sub_path[ancester_id]
    sub_path=sub_path[ancester_id:]
    sub_path.reverse()
    obj_path=obj_path[(ancester_id+1):]
    sdp_path=sub_path+obj_path
    for n in sdp_path:
        n.sdpdis = 0
    # for h in range(ancester_id+1, len(obj_path)):
    #     obj_path[h].sdpdis = 0
    return ancester,sdp_path,minsdplength,select_pair

def head_to_tree(head,deprel,subj_idx,obj_idx,entity_ids):
    """
    Convert a sequence of head indexes into a tree object.
    """
    #head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]
    subj_node=nodes[subj_idx[0]]
    obj_node=nodes[obj_idx[0]]
    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].rel = deprel[i]
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None

    spdlca,sdplength=getSDPAncestor(root,subj_node,obj_node)

    for s_idx in subj_idx:
        #     nodes[s_idx].subjdis = 0
        nodes[s_idx].sdpdis = 0
        # nodes[s_idx].domainid = 0

    for o_idx in obj_idx:
        #     nodes[o_idx].objdis = 0
        nodes[o_idx].sdpdis = 0
        # nodes[o_idx].domainid = 1
        #tree_to_dis(obj_node,None,0,0)

    domainid = 2
    for entity in entity_ids:
        if nodes[e_idx][0].sdpdis != 0:
            continue
        for e_idx in entity:
            nodes[e_idx].domainid = domainid
        domainid += 1

    domains=tree_to_dis(subj_node,None,0,domainid)

    return root,domains+1,sdplength
#

def head_to_treeEval(head,deprel,ners,pos,entity_ner,entity_pos,relpairs,build_mid):
# def head_to_tree(head,deprel, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    #head = head[:len_].tolist()
    root = None

    midhead=[-1 for _ in head]
    nodes = [Tree() for _ in head]
    pairlist=[]
    for pair in relpairs:
        subj=nodes[pair[0][0]]
        obj = nodes[pair[-1][0]]
        pairlist.append([subj,obj])
    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].rel = deprel[i]
        nodes[i].pos = pos[i]
        nodes[i].ner = ners[i]
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    spdlca,sdp_path,sdplength,pair_id=getSDPAncestorEval(root,pairlist)
    domains=0




    subj_nodes=pairlist[pair_id][0]
    obj_nodes=pairlist[pair_id][-1]
    pair=relpairs[pair_id]
    subj_idxs=relpairs[pair_id][0]
    obj_idxs=relpairs[pair_id][1]
    for s_idx in pair[0]:
        nodes[s_idx].sdpdis = 0
        nodes[s_idx].domainid = 0
    for o_idx in pair[-1]:
        nodes[o_idx].sdpdis = 0
        nodes[o_idx].domainid = 0


    # domainid=2
    # for entity in entity_ids:
    #     if nodes[e_idx][0].sdpdis!=0 and nodes[e_idx][0].domainid!=-1:
    #         continue
    #     for e_idx in entity:
    #         nodes[e_idx].domainid=domainid
    #     domainid+=1
    # domains=tree_to_dis(subj_nodes,None,0,domainid)
    # for pair in relpairs[pair_id]:
    # nodes[s_idx].subjdis = 0
    entities=[]
    dom_subj_id=None
    dom_obj_id=None
    i=0


    def inter(a, b):
        return not (len(set(a) & set(b)) == 0)

    while (i<len(head)):
        if ners[i] in entity_ner:
            e_ner=ners[i]
            entity=[]
            k=0
            while ((i+k)<len(head)):
                if ners[i+k] !=e_ner:
                    break
                else:
                    entity.append(i+k)
                    k=k+1
            i+=k
            if not(inter(entity,subj_idxs) or inter(entity,obj_idxs)):
                entities.append(entity)
        elif pos[i] in entity_pos and nodes[i].sdpdis==0:
            entities.append([i])
        i+=1

    # entity_chains=[]
    # for i,node in enumerate(sdp_path):
    #     if node.idx in entities:



    # for i in range(len(head)):
    #     if nodes[i].sdpdis==0 and (deprel[i] in no_pass):
    #         entities.add(i)
    #         if i in subj_idxs:
    #             dom_subj_id=i
    #         if i in obj_idxs:
    #             dom_obj_id=i
    entities.append(subj_idxs)
    dom_subj_id=subj_idxs
    entities.append(obj_idxs)
    dom_obj_id=obj_idxs

    # black_list=set()

    # for node in subj_idxs:
    #     next=head[node] - 1
    #     if next in subj_idxs:
    #         if next in entities:
    #             entities.remove(next)
    #         if next not in black_list:
    #             black_list.add(next)
    # add_entities=set(subj_idxs)-black_list
    # entities=entities.union(add_entities)
    #
    # black_list=set()
    # for node in obj_idxs:
    #     next=head[node] - 1
    #     if next in obj_idxs:
    #         if next in entities:
    #             entities.remove(next)
    #         if next not in black_list:
    #             black_list.add(next)
    #
    #
    # add_entities = set(obj_idxs) - black_list
    # entities = entities.union(add_entities)
    #
    # #tree_to_dis(obj_node,None,0,0)
    # if build_mid:
    #     for i in range(len(nodes)):
    #         if midhead[i]!=-1:
    #             continue
    #         else:
    #             cur = i
    #             nodestack=[cur]
    #             # if deprel[cur] in no_pass:
    #             #     h=head[cur]-1
    #             #     for node in nodestack:
    #             #         midhead[node] = h
    #             #     nodestack = [cur,h]
    #             #     cur=h
    #             while(True):
    #                 h = head[cur] - 1
    #                 if h==-1:
    #                     for node in nodestack:
    #                         midhead[node]=cur
    #                         nodes[node].depnode=cur
    #                     break
    #                 if h in entities:
    #                     for node in nodestack:
    #                         midhead[node] = h
    #                         nodes[node].depnode=h
    #                     if deprel[cur]==appos or (cur in subj_idxs and h in subj_idxs) or (cur in obj_idxs and h in obj_idxs):
    #                         nodestack=[cur,h]
    #                     else:
    #                         nodestack=[h]
    #                 elif midhead[h] != -1:
    #                     for node in nodestack:
    #                         midhead[node] = midhead[h]
    #                         nodes[node].depnode=h
    #                     break
    #                 else:
    #                     nodestack.append(h)
    #                 cur=h
    # domain_dict={}
    # for i,h in enumerate(midhead):
    #     if h not in domain_dict.keys():
    #         domid=len(domain_dict)
    #         domain_dict[h]=domid
    #     nodes[h].domainid=domain_dict[h]


    entity_chains = []
    # rid=root.idx
    # domain_subj=tree_to_dis(su,None,0,0)

    # searchqueue=[]
    # while len(queue):
    #     cur=queue[0]
    #     if cur not in subj_idxs:
    #         searchqueue.append(subj_idxs)
    #     else:

    sdpentitynodes=[]
    domsubj=sdp_path[0].idx
    for node in sdp_path:
        if node.idx not in subj_idxs and node.idx not in obj_idxs:
            obj = dfs(node, None, entity_pos, entity_ner)
            if obj is not None:
                subj_entity=[domsubj]
                obj_entity=[obj.idx]
                for ent in entities:
                    if domsubj in ent:
                        subj_entity=ent
                    if obj.idx in ent:
                        obj_entity=ent
                if subj_entity!=obj_entity:
                    entity_chains.append([subj_entity,obj_entity])
                    sdpentitynodes.append(node.idx)
                domsubj=obj.idx
    #sdpentitynodes.append(sdp_path[-1].idx)
    subj_entity=[domsubj]
    for ent in entities:
        if domsubj in ent:
            subj_entity=ent

    if subj_entity != obj_idxs:
        entity_chains.append([subj_entity,obj_idxs])

    domain=0
    labdom(sdp_path[0],None,set([domain]))
    for node in sdp_path:
        if node.idx in sdpentitynodes:
            domain+=1
            tempd=set((domain-1,domain))
        else:
            tempd=set([domain])
        labdom(node,None,tempd)
    labdom(sdp_path[-1],None,set([domain]))
    sdp_domain=list(range(domain+1))
    # subj_entity=nodes[dom_subj_id]
    # dom_subj_entity=dom_subj_id
    # raw_entities=copy.deepcopy(entities)
    # while(dom_subj_entity is not None):
    #     next_subj_entity=None
    #     next_list=[]
    #     for subj_i in dom_subj_entity:
    #         subj_n=nodes[subj_i]
    #         for child in subj_n.children:
    #             if child.idx not in dom_subj_entity:
    #                 next_list.append(child)
    #         if (subj_n.parent is not None) and (subj_n.parent.idx not in dom_subj_entity):
    #             next_list.append(subj_n.parent)
    #     # next_list+=next_subj.children
    #     entities.remove(dom_subj_entity)
    #     # if next_subj.parent is not None:
    #     #     next_list.append(next_subj.parent)
    #     for next in next_list:
    #         for subj_i in dom_subj_entity:
    #             nodes[subj_i].domainids.add(domains)
    #         _,dom_obj_entity=tree_to_dis(next,entities,0,domains)
    #         if dom_obj_entity is not None:
    #             chain=[dom_subj_entity,dom_obj_entity]
    #             if chain in entity_chains:
    #                 domains=domains-1
    #             else:
    #                 entity_chains.append(chain)
    #                 sdp_domain.append(domains)
    #                 next_subj_entity=dom_obj_entity
    #         domains=domains+1
    #     dom_subj_entity=next_subj_entity
    #
    #
    # for entity in raw_entities:
    #     entity_domains=set()
    #     for n in entity:
    #         entity_domains=entity_domains.union(nodes[n].domainids)
    #     for m in entity:
    #         nodes[m].domainids=nodes[m].domainids.union(entity_domains)

    return root,domain+1,sdplength,relpairs[pair_id],midhead,entity_chains,sdp_domain
# def head_to_treeEval(head,deprel,subj_ids,obj_ids,build_mid):
# # # def head_to_tree(head,deprel, len_):
#     """
#     Convert a sequence of head indexes into a tree object.
#     """
#     #head = head[:len_].tolist()
#     no_pass=constant.no_pass
#     appos=constant.appos
#     root = None
#     midhead=[-1 for _ in head]
#     nodes = [Tree() for _ in head]
#     pairlist=[]
#     relpairs=[]
#     for subj_id in subj_ids:
#         for obj_id in obj_ids:
#             subj=nodes[subj_id[0]]
#             obj = nodes[obj_id[0]]
#             if subj==obj:
#                 continue
#             relpairs.append([subj_id,obj_id])
#             pairlist.append([subj,obj])
#     for i in range(len(nodes)):
#         h = head[i]
#         nodes[i].idx = i
#         nodes[i].rel = deprel[i]
#         nodes[i].dist = -1  # just a filler
#         if h == 0:
#             root = nodes[i]
#         else:
#             nodes[h - 1].add_child(nodes[i])
#
#     assert root is not None
#     spdlca,sdplength,pair_id=getSDPAncestorEval(root,pairlist)
#     subj_nodes=pairlist[pair_id][0]
#     obj_nodes=pairlist[pair_id][-1]
#     domains=tree_to_dis(subj_nodes,None,0,2)
#     # for pair in relpairs[pair_id]:
#     # nodes[s_idx].subjdis = 0
#     pair=relpairs[pair_id]
#     for s_idx in pair[0]:
#         nodes[s_idx].sdpdis = 0
#         nodes[s_idx].domainid = 0
#     for o_idx in pair[-1]:
#         nodes[o_idx].sdpdis = 0
#         nodes[o_idx].domainid = 1
#     #tree_to_dis(obj_node,None,0,0)
#     if build_mid:
#         for i in range(len(nodes)):
#             if midhead[i]!=-1:
#                 continue
#             else:
#                 cur = i
#                 nodestack=[cur]
#                 # if deprel[cur] in no_pass:
#                 #     h=head[cur]-1
#                 #     for node in nodestack:
#                 #         midhead[node] = h
#                 #     nodestack = [cur,h]
#                 #     cur=h
#                 while(True):
#                     h = head[cur] - 1
#                     if h==-1:
#                         for node in nodestack:
#                             midhead[node]=cur
#                         break
#                     if deprel[h] in no_pass:
#                         for node in nodestack:
#                             midhead[node] = h
#                         if deprel[cur]==appos:
#                             nodestack=[cur,h]
#                         else:
#                             nodestack=[h]
#                     elif midhead[h] != -1:
#                         for node in nodestack:
#                             midhead[node] = midhead[h]
#                         break
#                     else:
#                         nodestack.append(h)
#                     cur=h
#return root,domains+1,sdplength,relpairs[pair_id],midhead

def dfs(cur,src,pos,ner):
    if cur.pos in pos or cur.ner in ner:
        return cur
    entity=None
    for child in cur.children:
        if child!=src and child.sdpdis!=0:
            entity=dfs(child,cur,pos,ner)
            if entity is not None:
                return entity
    if cur.parent is not None and cur.parent!=src and cur.parent.sdpdis!=0:
        entity = dfs(cur.parent, cur, pos, ner)
    return entity

def labdom(cur,src,domains):
    cur.domainids=cur.domainids.union(domains)
    for child in cur.children:
        if child !=src and child.sdpdis!=0:
            labdom(child,cur,domains)
    if cur.parent is not None and cur.parent!=src and cur.parent.sdpdis!=0:
        labdom(cur.parent,cur,domains)


def tree_to_dis(cur,src,sdpdis,domain):
    if len(cur.domainids)!=0:
        return domain,None
    cur.domainids.add(domain)
    for entity in src:
        if cur.idx in entity:
            return domain,entity

    # if cur.sdpdis == constant.MAX_DIS:
    #     cur.sdpdis=sdpdis
    # if cur.sdpdis!=0:
    #     cur.domainid=domain

    # if cur.domainid==-1:
    #     cur.domainid=domain
    endnode=None
    if (len(cur.children)!=0):
        for child in cur.children:
            if len(child.domainids)!=0:
                continue
            else:
                domain,cendnode=tree_to_dis(child, src, sdpdis, domain)
                if endnode is None:
                    endnode=cendnode
    if cur.parent is not None:
        if len(cur.parent.domainids)==0:
            domain,cendnode=tree_to_dis(cur.parent, src, sdpdis, domain)
            if endnode is None:
                endnode=cendnode
    # if (cur.parent is not None) and (cur.parent!=src):
    #     if cur. and cur.parent.domainid==-1:
    #         domain=min(25,domain+1)
    #         domain=tree_to_dis(cur.parent,cur,sdpdis+1,domain+1)
    #     else:
    #         domain = tree_to_dis(cur.parent, cur, sdpdis + 1, domain)
    return domain,endnode



def tree_to_adj(sent_len,domains, tree,entity_chains,sdp_domains,directed=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    domain=np.zeros((sent_len,domains), dtype=np.float32)
    # domain_id=np.zeros((sent_len,domains),dtype=np.float32)
    # redomain_id=np.zeros((sent_len,domains),dtype=np.float32)
    # sdp_domain=np.zeros((domains,domains),dtype=np.float32)
    domain_subj=np.zeros((sent_len,domains),dtype=np.float32)
    domain_obj=np.zeros((sent_len,domains),dtype=np.float32)
    sdp_domain=np.zeros((domains,domains),dtype=np.float32)
    rel =np.zeros_like(ret,dtype=np.int64)
    k=constant.MAX_DIS
    #subj_pos=np.ones(sent_len,dtype=np.int64)*k
    #obj_pos=np.ones(sent_len,dtype=np.int64)*k
    #sdp_dis=np.ones(sent_len,dtype=np.int64)*k
    queue = [tree]
    idx = []
    seq_len=1
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        # subj_pos[t.idx]=min(k,t.subjdis)
        # obj_pos[t.idx]=min(k,t.objdis)
        # sdp_dis[t.idx]=min(k,t.sdpdis)
        for d in t.domainids:
            domain[t.idx,d]=1
        for c in t.children:
            ret[t.idx, c.idx] = 1
            rel[t.idx,c.idx]= t.rel
            # if t.domainid!=c.domainid:
            #     domain_id[c.idx,c.domainid]=1
            #     redomain_id[c.idx,t.domainid]=1
        queue += t.children
        seq_len+=len(t.children)
    for i in range(len(sdp_domains)):
        for j in entity_chains[i][0]:
            domain_subj[j][sdp_domains[i]] = 1
        for k in entity_chains[i][-1]:
            domain_obj[k][sdp_domains[i]] = 1
    depmap=ret.T.copy()
    rel=depmap.copy()
    # if(sent_len>seq_len):
    #      depmap[seq_len:,0]=1
    depmap[tree.idx,0]=1
    if not directed:
        ret = ret + ret.T
    for i,d in enumerate(sdp_domains):
        sdp_domain[i][d]=1
    #return depmap,ret,rel,subj_pos,obj_pos,sdp_dis
    # eye=np.identity(sent_len)
    # ndom=domain.dot(domain.T)
    # mulrel=rel+eye
    # resrel=mulrel
    # while True:
    #     newresrel=1*(resrel.dot(mulrel)!=0)
    #     if (newresrel==resrel).all():
    #         break
    #     resrel=newresrel
    # resrel=1*((resrel*ndom+rel)>0)-eye
    return depmap,ret,rel,rel,domain,sdp_domain,domain_subj,domain_obj,