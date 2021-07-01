import os
import argparse
import re
from collections import defaultdict
import logging


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
		# 	continue
		else:
			continue
			# raise RuntimeError('invalid brat data: {}'.format(each))
	return ners, rels


def __load_ner_dict(data):
	#convert data dict to list, each element will be combine with its file name
	new_data = []
	for k, v in data.items():
		for each in v:
			each.append(k)
			new_data.append(each)
	return new_data


def __match(en, others):
	#TODO
	return False


def generate_metrics(src, ref):
	import numpy as np
	# based on method describe here: http://aclweb.org/anthology/W16-1708
	ref_data = __load_ner_dict(ref)
	src_data = __load_ner_dict(src)
	dim = len(ref_data)
	metric = np.zeros((dim, dim))
	for idx, en in enumerate(ref_data):
		if __match(en, src_data):
			#TODO
			pass


def calc_kappa_m(tp, fp, fn, tn):
	# based on cohen kappa definition
	p0 = (tp + tn) / (tp + tn + fp + fn)
	py = ((tp + fn) / (tp + tn + fp + fn)) * ((tp + fp) / (tp + tn + fp + fn))
	pn = ((fp + tn) / (tp + tn + fp + fn)) * ((fn + tn) / (tp + tn + fp + fn))
	pe = py + pn
	return (p0-pe)/(1-pe)


def calc_kappa_sk(src, ref, labels=None, weights=None, sample_weight=None):
	from sklearn.metrics import cohen_kappa_score

	return cohen_kappa_score(src, ref, labels=labels, weights=weights, sample_weight=sample_weight)


def extract_entities(src, ref):
	src_ner_data = defaultdict(list)
	ref_ner_data = defaultdict(list)

	for each in os.listdir(src):
		if each.endswith(".ann"):
			src_ner_data[each].extend(read_brat(os.path.join(src, each))[0])

	for each in os.listdir(ref):
		if each.endswith(".ann"):
			ref_ner_data[each].extend(read_brat(os.path.join(ref, each))[0])

	return src_ner_data, ref_ner_data


def __mapping(tokens, txt):
        offset_original = 0
        token_offsets = []

        for each in tokens:
            try:
                index = txt.index(each)
            except ValueError as ex:
                logger.error("the {} cannot be find in original text.".format(each))
                continue

            offset_original += index
            original_start = offset_original
            tk_len = len(each)
            new_pos = index + tk_len
            offset_original += tk_len
            original_end = offset_original
            txt = txt[new_pos:]
            token_offsets.append((original_start, original_end))
       
        return token_offsets


def generate_bio(fid, fdir):
	bio = []
	#['MRN:', 0, 4]
	tok_pos = []
	#['T32', 'OSS', 3596, 3614, 'Abnormal mammogram']
	ann_ner = read_brat(os.path.join(fdir, fid+".ann"))[0]
	ann_ner = sorted(ann_ner, key=lambda x: x[2]) 

	with open(os.path.join(fdir, fid+".txt"), "r") as f:
		text = f.read().strip()
	tokens = text.split(" ") 
	offsets = __mapping(tokens, text)
	for each in zip(tokens, offsets):
		tok_pos.append([each[0], each[1][0], each[1][1]])

	ner_iter = iter(ann_ner)
	en = next(ner_iter, None)
	for idx, tok in enumerate(tok_pos):
		if not en:
			tok.append('O')
		else:
			ts, te = tok[1], tok[2]
			ent, ens, ene = en[1], en[2], en[3]
			if ts < ens and te < ene:
				tok.append('O')
			elif ts == ens:
				tok.append(ent)
				if te >= ene:
					en = next(ner_iter, None)
			elif ts > ens and te < ene:
				tok.append(ent)
			elif ts > ens and te == ene:
				tok.append(ent)
				en = next(ner_iter, None)
			else:
				#overlap case
				tok.append('O')
				en = next(ner_iter, None)
		bio.append(tok)

	return bio

def create_file_ids(fdir):
	fids = []
	for each in os.listdir(fdir):
		if each.endswith(".txt"):
			fids.append(each.split(".")[0])
	return fids


def annotation2tags(src, ref):
	sfid = create_file_ids(src)
	rfid = create_file_ids(ref)
	assert sfid == rfid, "source and reference should have same files but {} are different".format(set(rfid).symmetric_difference(set(sfid)))

	ranns = []
	sanns = []
	for fid in rfid:
		ranns.extend(generate_bio(fid, ref))
		sanns.extend(generate_bio(fid, src))

	ranns = [each[-1] for each in ranns]
	sanns = [each[-1] for each in sanns]

	# print(ranns)

	return sanns, ranns 

def annotation_agreement(src, ref, kappa_method):
	# logging.basicConfig(filename='kappa.log',level=logging.INFO, format='%(asctime)s - %(message)s')
	logger = logging.getLogger('calculate brat annotation kappa') 

	if kappa_method == "1":
		src_ner_data, ref_ner_data = extract_entities(src, ref)
		k = generate_metrics(src_ner_data, ref_ner_data)
	elif kappa_method == "2":
		src_ner_tags, ref_ner_tags = annotation2tags(src, ref)
		k = calc_kappa_sk(src_ner_tags, ref_ner_tags)
	else:
		raise NotImplementedError('only methods 1 and 2 are implemented')
	
	logger.info("kappa is {}".format(k))


def list2type_dict(l):
	# each term in list should have type as the second element
	##['T32', 'OSS', 3596, 3614, 'Abnormal mammogram']
	d = dict()
	for each in l:
		t = each[1]
		if t in d:
			d[t] += 1
		else:
			d[t] = 1
	return d


def entity_relation_count(src):
	# logging.basicConfig(filename='brat__entity_relation_count.log',level=logging.INFO, format='%(asctime)s - %(message)s')
	logger = logging.getLogger('brat_count')

	enss = []
	relss = []

	for brat_file in os.listdir(src): 
		if brat_file.endswith(".ann"):
			ens, rels = read_brat(os.path.join(src, brat_file))
			enss.extend(ens)
			relss.extend(rels)

	logger.info("total number of entities {}; total number of relations".format(len(enss), len(relss)))

	en_dict = list2type_dict(enss)
	rel_dict = list2type_dict(relss)

	template = "entities count by type:\n"
	for k, v in en_dict.items():
		template = "{} {}: {}\n".format(template, k, v)
	template += "relations count by type:\n"
	for k, v in rel_dict.items():
		template = "{} {}: {}\n".format(template, k, v)

	logger.info(template)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", default="1", help="modes for task:\n1 == kappa\n 2 == entities and relation numbers\n 3==")
	parser.add_argument("-s", "--src", help="source directory of all the brat annotation files", required=True)
	parser.add_argument("-r", "--ref", default=None, help="directory of all the brat annotation files as references")
	parser.add_argument("-k", "--kappa_method", default="2", help="methods used for calculate kappa: 1==entity level; 2==token level")
	parser.add_argument("-l", "--logging_filename", default=None, help="filename for logging")
	
	args = parser.parse_args()

	if args.logging_filename:
		logging.basicConfig(filename=args.logging_filename,level=logging.INFO, format='%(asctime)s - %(message)s')
	else:
		logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
		
	if args.mode == "1":
		annotation_agreement(args.src, args.ref, args.kappa_method)
	elif args.mode == "2":
		entity_relation_count(args.src)
	else:
		# for each in os.listdir(args.src):
		# 	if each.endswith(".ann"):
		# 		print(read_brat(os.path.join(args.src, each)))
		raise NotImplementedError('TODO read brat info')