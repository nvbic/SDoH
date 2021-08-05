#usepipeline trans to relation extracttion used data
from pathlib import Path
import pickle as pkl
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os
def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)
        
def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data

def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt
import sys
sys.path.append("/home/zehao.yu/workspace/py3/NLPreprocessing/")
from annotation2BIO import pre_processing, read_annotation_brat, generate_BIO
MIMICIII_PATTERN = "\[\*\*|\*\*\]"
def parse_brat_ner(brat_data):
    # assert re.match("^T[0-9]+\t[A-Za-z]+ [0-9]+ [0-9]+\t\.+$", brat_data), "invalid brat format for {}".format(brat_data)
    info = brat_data.split("\t")
    idx = info[0]
    text = info[2]
    tse = info[1]
    if ";" in tse:
        ii = tse.split(" ")
        return [idx, ii[0], " ".join(ii[1:-1]), ii[-1], text]
    else:
        tag, s, e = tse.split(" ")
        return [idx, tag, int(s), int(e), text]


def parse_brat_rel(brat_data):
    # assert re.match("^R[0-9]+\t[A-Za-z]+-[A-Za-z]+ Arg1:T[0-9]+ Arg2:T[0-9]+$", brat_data), "invalid brat format for {}".format(brat_data)
    info = brat_data.split("\t")
    idx = info[0]
    tag, arg1, arg2 = info[1].split(" ")
    arg1 = arg1.split(":")[-1]
    arg2 = arg2.split(":")[-1]
    return [idx, tag, arg1, arg2]


def read_brat(file_name):
    ners = []
    rels = []

    with open(file_name, "r") as f:
        cont = f.read().strip()
    
    if not cont:
        return ners, rels

    # process ner and relation
    for each in cont.split("\n"):
        if each.startswith("T"):
            ners.append(parse_brat_ner(each))
        elif each.startswith("R"):
            rels.append(parse_brat_rel(each))
        # elif each.startswith("#"):
        #     continue
        else:
            continue
            # raise RuntimeError('invalid brat data: {}'.format(each))
    return ners, rels
def get_all_rel_types(root, valid_comb):
    root = Path(root)
    rtt = []
    for ann_fn in root.glob("*.ann"):
#         _, rels = read_brat(ann_fn)
        _, _, rels = read_annotation_brat(ann_fn)
        #('Strength-Drug', 'T3', 'T2')
        for rel in rels:
#             rtype = rel[1]
            rtype = rel[0]
            if tuple(rtype.split("-")) not in valid_comb:
                continue
            rtt.append(rtype)
    rs, rc = set(rtt), Counter(rtt).most_common()
    r2i = {NEG_REL: '0'}
    for idx, rt in enumerate(rs):
        r2i[rt] = str(idx + 1)
    i2r = {v: k for k, v in r2i.items()}
    return r2i, i2r, rc
def create_entity_to_sent_mapping(nnsents, entities, idx2e):
    loc_ens = []
    
    ll = len(nnsents)
    mapping = defaultdict(list)
    for idx, each in enumerate(entities):
        en_label = idx2e[idx]
        en_s = each[2][0]
        en_e = each[2][1]
        new_en = []
        
        i = 0
        while i < ll and nnsents[i][1][0] < en_s:
            i += 1
        s_s = nnsents[i][1][0]
        s_e = nnsents[i][1][1]

        if en_s == s_s:
            mapping[en_label].append(i)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
        else:
            mapping[en_label].append(i)
            print("first index not match ", each)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
    return mapping

def get_permutated_relation_pairs(eid2idx):
    all_pairs = []
    all_ids = [k for k, v in eid2idx.items()]
    for e1, e2 in permutations(all_ids, 2):
        all_pairs.append((e1, e2))
    return all_pairs
def validate_rels(rels, valid):
    nrels = []
    for rel in rels:
        rtype = rel[0]
        if tuple(rtype) not in valid:
            print("invalid: ", rel)
            continue
        nrels.append(rel)
    return nrels


def check_tags(s1, s2):
    assert EN1_START in s1 and EN1_END in s1, f"tag error: {s1}"
    assert EN2_START in s2 and EN2_END in s2, f"tag error: {s2}"


def format_relen(en, rloc, nsents):
    if rloc == 1:
        spec1, spec2 = EN1_START, EN1_END
    else:
        spec1, spec2 = EN2_START, EN2_END
    sn1, tn1 = en[0][3]
    sn2, tn2 = en[-1][3]
    target_sent = nsents[sn1]
    target_sent = [each[0] for each in target_sent]
    ors =  " ".join(target_sent)
    
    if sn1 != sn2:
#         print("[!!!Warning] The entity is not in the same sentence\n", en)
        tt = nsents[sn2]
        tt = [each[0] for each in tt]
        target_sent.insert(tn1, spec1)
        tt.insert(tn2+1, spec2)
        target_sent = target_sent + tt
#         print(target_sent)
    else:
        target_sent.insert(tn1, spec1)
        target_sent.insert(tn2+2, spec2)
    
    fs = " ".join(target_sent)
    
    return sn1, sn2, fs, ors


def gene_true_relations(rels, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=None):
    true_pairs = set()
    pos_samples = []
    
    for rel in rels:
        rel_type = rel[0]
        enid1, enid2 = rel[1:]
        enbs1, enbe1 = mappings[enid1]
        en1 = nnsents[enbs1: enbe1+1]
        si1, sii1, fs1, ors1 = format_relen(en1, 1, nsents)
        enbs2, enbe2 = mappings[enid2]
        en2 = nnsents[enbs2: enbe2+1]
        si2, sii2, fs2, ors2 = format_relen(en2, 2, nsents)
        sent_diff = abs(si1 - si2)
        
        en1t = en1[0][-1].split("-")[-1]
        en2t = en2[0][-1].split("-")[-1]
        
#         print(abs(si1 - si2), abs(sii1 - sii2), abs(sii1 - si2), abs(si1 - sii2))
        
        true_pairs.add((enid1, enid2))
        
        if (en1t, en2t) not in valid_comb:
            continue
        
        if sent_diff <= CUTOFF:
            check_tags(fs1, fs2)
            assert (en1t, en2t) in valid_comb, f"{en1t} {en2t}"
            if fid:
                pos_samples.append((sent_diff, rel_type, fs1, fs2, en1t, en2t, enid1, enid2, fid))
#                 pos_samples.append((sent_diff, "pos", fs1, fs2, en1t, en2t, enid1, enid2, fid))
            else:
                pos_samples.append((sent_diff, rel_type, fs1, fs2, en1t, en2t, enid1, enid2))
#                 pos_samples.append((sent_diff, "pos", fs1, fs2, en1t, en2t, enid1, enid2))
#         print(sent_diff, rel_type, fs1, fs2, ors1, ors2)
    
    return pos_samples, true_pairs
        

def gene_neg_relation(perm_pairs, true_pairs, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=None):
    neg_samples = []
    for each in perm_pairs:
        enid1, enid2 = each
        
        # not in true relation
        if (enid1, enid2) in true_pairs:
            continue
        
        enc1 = ens[e2i[enid1]]
        enc2 = ens[e2i[enid2]]
    
        #('Metoprolol succinate', 'Drug', (14660, 14680))
        
        enbs1, enbe1 = mappings[enid1]
        en1 = nnsents[enbs1: enbe1+1]
        si1, sii1, fs1, ors1 = format_relen(en1, 1, nsents)
        enbs2, enbe2 = mappings[enid2]
        en2 = nnsents[enbs2: enbe2+1]
        si2, sii2, fs2, ors2 = format_relen(en2, 2, nsents)
        sent_diff = abs(si1 - si2)
        
        en1t = en1[0][-1].split("-")[-1]
        en2t = en2[0][-1].split("-")[-1]
        
#         print((enc1[1], enc2[1]), (en1t, en2t))
        
        if (en1t, en2t) not in valid_comb:
            continue
        
        if sent_diff <= CUTOFF:
            check_tags(fs1, fs2)
            assert (en1t, en2t) in valid_comb, f"{en1t} {en2t}"
            if fid:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2, fid))
            else:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2))
    
    return neg_samples

    
def create_training_samples(file_path, valids=None, valid_comb=None):
    fids = []
    root = Path(file_path)
    
    dpos = defaultdict(list)
    dneg = defaultdict(list)
    
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem+".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
#         pre_txt = pre_txt.split("\n")
        e2i, ens, rels = read_annotation_brat(ann_fn)
#         ens, rels = read_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
#         rels = validate_rels(rels, n2c2_valid_comb)
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
#         print(nsents)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)

        pos_samples, true_pairs = gene_true_relations(
            rels, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        perm_pairs = get_permutated_relation_pairs(e2i)
        neg_samples = gene_neg_relation(
            perm_pairs, true_pairs, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        
        for pos_sample in pos_samples:
            dpos[pos_sample[0]].append(pos_sample)
        for neg_sample in neg_samples:
            dneg[neg_sample[0]].append(neg_sample)
#         break
        
    return dpos, dneg


def create_test_samples(file_path, valids=None, valid_comb=None):
    #create a separate mapping file
    rel_mappings = []
    #
    fids = []
    root = Path(file_path)
    preds = defaultdict(list)
    
    g_idx = 0
    
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem + ".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
        e2i, ens, _ = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)
        
        perm_pairs = get_permutated_relation_pairs(e2i)
        pred = gene_neg_relation(perm_pairs, set(), mappings, ens, e2i, nnsents, nsents, valid_comb, fid=txt_fn.stem)
        for idx, pred_s in enumerate(pred):
            preds[pred_s[0]].append(pred_s)
#             rel_mappings.append((g_idx, *pred_s[-3:]))
#             g_idx += 1
            
    return preds #rel_mappings
def to_tsv(data, fn):
    header = "\t".join([str(i+1) for i in range(len(data[0]))])
    with open(fn, "w") as f:
        f.write(f"{header}\n")
        for each in data:
            d = "\t".join([str(e) for e in each])
            f.write(f"{d}\n")


def to_5_cv(data, ofd):
    if not os.path.isdir(ofd):
        os.mkdir(ofd)
    
    np.random.seed(13)
    np.random.shuffle(data)
    
    dfs = np.array_split(data, 5)
    a = [0,1,2,3,4]
    for each in combinations(a, 4):
        b = list(set(a) - set(each))[0]
        n = dfs[b]
        m = []
        for k in each:
            m.extend(dfs[k])
        if not os.path.isdir(os.path.join(ofd, f"sample{b}")):
            os.mkdir(os.path.join(ofd, f"sample{b}"))
        
        to_tsv(m, os.path.join(ofd, f"sample{b}", "train.tsv"))
        to_tsv(n, os.path.join(ofd, f"sample{b}", "dev.tsv"))


def all_in_one(*dd, dn="2018n2c2", do_train=True):
    data = []
    for d in dd:
        for k, v in d.items():
            for each in v:
#                 data.append(each[1:])
                data.append(each)
    
    output_path = f"/data/userData/zehao.yu/sdoh/data/{dn}_aio_th{CUTOFF}"
    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)
    
    if do_train:
        to_tsv(data, p/"train.tsv")
#         to_5_cv(data, p.as_posix())
    else:
        to_tsv(data, p/"test.tsv")
    

def all_in_unique(*dd, dn="2018n2c2", do_train=True):
    for idx in range(CUTOFF+1):
        data = []
        for d in dd:
            for k, v in d.items():
                for each in v:
                    if k == idx:
                        data.append(each[1:])
        
        output_path = f"/data/userData/zehao.yu/sdoh/data/{dn}_aiu_th{CUTOFF}"
        p = Path(output_path) / f"cutoff_{idx}"
        p.mkdir(parents=True, exist_ok=True)
        if do_train:
            to_tsv(data, p/"train.tsv")
#             to_5_cv(data, p.as_posix())
        else:
            to_tsv(data, p/"test.tsv")

            
def partial_unique(*dd, dn="2018n2c2", do_train=True):
    within = []
    cross = []
    
    for d in dd:
        for k, v in d.items():
            for each in v:
                if k == 0:
                    within.append(each[1:])
                else:
                    cross.append(each[1:])
    
    output_path = f"./data/{dn}_pu_th{CUTOFF}"
    p = Path(output_path)
    p1 = p / "within"
    p2 = p / "cross"
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)
    
    if do_train:
        to_tsv(within, p1/"train.tsv")
#         to_5_cv(within, p1.as_posix())
        to_tsv(cross, p2/"train.tsv")
#         to_5_cv(cross, p2.as_posix())
    else:
        to_tsv(within, p1/"test.tsv")
        to_tsv(cross, p2/"test.tsv")
def extract_only_entity(input_path, output_path):
    pi = Path(input_path)
    po = Path(output_path)
    po.mkdir(exist_ok=True, parents=True)
    for fid in pi.glob("*.ann"):
        ofn = po / fid.name
        with open(fid, "r") as f1, open(ofn, "w") as f2:
            for line in f1.readlines():
                if line.startswith("T"):
                    f2.write(line)
EN1_START = "[s1]"
EN1_END = "[e1]"
EN2_START = "[s2]"
EN2_END = "[e2]"
NEG_REL = "NonRel"
CUTOFF = 1
sdoh_valid_comb = {
        ('Tobacco_use', 'Substance_use_status'), ('Substance_use_status', 'Smoking_type'),
        ('Substance_use_status', 'Smoking_freq_ppd'), ('Substance_use_status', 'Smoking_freq_py'), 
        ('Substance_use_status', 'Smoking_freq_qy'), ('Substance_use_status', 'Smoking_freq_sy'),
        ('Substance_use_status', 'Smoking_freq_other'), ('Alcohol_use', 'Substance_use_status'),
        ('Substance_use_status', 'Alcohol_freq'), ('Substance_use_status', 'Alcohol_type'), 
        ('Substance_use_status', 'Alcohol_other'), ('Drug_use', 'Substance_use_status'),
        ('Substance_use_status', 'Drug_freq'), ('Substance_use_status', 'Drug_type'),('Substance_use_status', 'Drug_other'), ('Sex_act', 'Sdoh_status'),
        ('Sex_act', 'Partner'), ('Sex_act', 'Protection'), 
        ('Sex_act', 'Sex_act_other'), ('Occupation', 'Employment_status'),
        ('Occupation', 'Employment_location'), ('Gender', 'Sdoh_status'),('Social_cohesion', 'Social_method'), ('Social_method', 'Sdoh_status'),
        ('Physical_act', 'Sdoh_status'), ('Physical_act', 'Sdoh_freq'), 
        ('Living_supply', 'Sdoh_status'), ('Abuse', 'Sdoh_status'),
        ('Transportation', 'Sdoh_status'), ('Health_literacy', 'Sdoh_status'),
        ('Financial_constrain', 'Sdoh_status'), ('Social_cohesion', 'Sdoh_status'),
        ('Social_cohesion', 'Sdoh_freq'), ('Gender', 'Sdoh_status'), 
        ('Race', 'Sdoh_status'), ('Ethnicity', 'Sdoh_status'),
        ('Living_Condition', 'Sdoh_status')
    }
entp2rel = {
        ('Tobacco_use', 'Substance_use_status'):'Tobacco_use-Substance_use_status', 
        ('Substance_use_status', 'Smoking_type'):'Substance_use_status-Smoking_type',
        ('Substance_use_status', 'Smoking_freq_ppd'):'Substance_use_status-Smoking_freq', ('Substance_use_status', 'Smoking_freq_py'):'Substance_use_status-Smoking_freq', 
        ('Substance_use_status', 'Smoking_freq_qy'):'Substance_use_status-Smoking_freq', ('Substance_use_status', 'Smoking_freq_sy'):'Substance_use_status-Smoking_freq',
        ('Substance_use_status', 'Smoking_freq_other'):'Substance_use_status-Smoking_freq', ('Alcohol_use', 'Substance_use_status'):'Alcohol_use-Substance_use_status',
        ('Substance_use_status', 'Alcohol_freq'):'Substance_use_status-Alcohol_freq',
        ('Substance_use_status', 'Alcohol_type'):'Substance_use_status-Alcohol_type', 
        ('Substance_use_status', 'Alcohol_other'):'Substance_use_status-Alcohol_other',
        ('Drug_use', 'Substance_use_status'):'Drug_use-Substance_use_status',
        ('Substance_use_status', 'Drug_freq'):'Substance_use_status-Drug_freq',
        ('Substance_use_status', 'Drug_type'):'Substance_use_status-Drug_type',
        ('Substance_use_status', 'Drug_other'):'Substance_use_status-Drug_other',
        ('Sex_act', 'Sdoh_status'):'Sex_act-Sdoh_status',
        ('Sex_act', 'Partner'):'Sex_act-Partner', 
        ('Sex_act', 'Protection'):'Sex_act-Protection', 
        ('Sex_act', 'Sex_act_other'):'Sex_act-Sex_act_other', 
        ('Occupation', 'Employment_status'):'Occupation-Employment_status',
        ('Occupation', 'Employment_location'):'Occupation-Employment_location',
        ('Gender', 'Sdoh_status'):'Gender-Sdoh_status',
        ('Social_cohesion', 'Social_method'):'Social_cohesion-Social_method',
        ('Social_method', 'Sdoh_status'):'Social_method-Sdoh_status',
        ('Physical_act', 'Sdoh_status'):'Physical_act-Sdoh_status', 
        ('Physical_act', 'Sdoh_freq'):'Physical_act-Sdoh_freq', 
        ('Living_supply', 'Sdoh_status'):'Living_supply-Sdoh_status', 
        ('Abuse', 'Sdoh_status'):'Abuse-Sdoh_status',
        ('Transportation', 'Sdoh_status'):'Transportation-Sdoh_status', 
        ('Health_literacy', 'Sdoh_status'):'Health_literacy-Sdoh_status',
        ('Financial_constrain', 'Sdoh_status'):'Financial_constrain-Sdoh_status', 
        ('Social_cohesion', 'Sdoh_status'):'Social_cohesion-Sdoh_status',
        ('Social_cohesion', 'Sdoh_freq'):'Social_cohesion-Sdoh_freq', 
        ('Gender', 'Sdoh_status'):'Gender-Sdoh_status', 
        ('Race', 'Sdoh_status'):'Race-Sdoh_status', 
        ('Ethnicity', 'Sdoh_status'):'Ethnicity-Sdoh_status',
        ('Living_Condition', 'Sdoh_status'):'Living_Condition-Sdoh_status'
    }
sdoh_valid=list(entp2rel.keys())
test_root='/home/zehao.yu/workspace/py3/SDoH/res/bert_final_formatted_output'
CUTOFF=1
#     dpos, dneg = create_training_samples(train_dev_root,dr_valid,dr_valid_comb_en)
#     all_in_one(dpos, dneg, dn="dr_relation", do_train=True)
#     partial_unique(dpos, dneg, dn="dr_relation", do_train=True)
#     all_in_unique(dpos, dneg, dn="dr_relation", do_train=True)
# dpos, dneg = create_training_samples(train_dev_root,sdoh_valid,sdoh_valid_comb)   
preds = create_test_samples(test_root,sdoh_valid,sdoh_valid_comb)
all_in_one(preds, dn="sdoh", do_train=False)
