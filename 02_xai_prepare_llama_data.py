from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
import en_core_web_sm
from nltk import ngrams
from utils.multiwoz import dbPointer
import ipdb
import json
import random
import os
import pickle

from transformers import GPT2Tokenizer, AutoTokenizer
llama3_tknzer = AutoTokenizer.from_pretrained('meta-llama/meta-llama-3-8B-instruct')

multiwoz_data = json.load(open('resources/multi-woz/lex.json', 'r'))
save_dir = './resources/llama3'
os.makedirs(save_dir, exist_ok=True)

for split in ['train', 'val', 'test']:

    opt = ArgsParser().parse()
    opt.use_knowledge = True
    opt.use_action = True
    opt.context_knowledge = True
    opt.lexical = True

    data = MultiWozDataset(opt, split=split, shuffle=False)

    opt_delex = ArgsParser().parse()
    data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)

    history_raw_new = []
    belief_raw_new = []
    belief_raw_none_new = []
    action_raw_new = []
    output_raw_new = []
    output_raw_delex_new = []
    db_search_raw = []
    db_nmatch_raw = []
    
    if split == 'test':
        test_dict = {}

    lex_dict = {}
    delex_dict = {}
    for d in data:
        lex_dict[d['name']] = d

    for d in data_delex:
        delex_dict[d['name']] = d

    for key in lex_dict:
        d_lex = lex_dict[key]
        d_delex = delex_dict[key]
        inp = d_lex['input_raw']
        out = d_lex['target_raw']
        out_delex = d_delex['target_raw']
        db_data = d_lex['db']
        goal = multiwoz_data[key]['goal']

        for i, (usr, sys) in enumerate(zip(inp, out)):
            if i == 0:
                history_new = 'user\n\n{}'.format(usr)
            else:
                tmp_new = []
                for k in range(i):
                    tmp_new.append('user\n\n' + inp[k])
                    tmp_new.append('assistant\n\n' + out[k])

                tmp_new.append('user\n\n'+usr)
                history_new = '\n\n'.join(tmp_new)

            sys_delex = out_delex[i]
            history_raw_new.append(history_new)
            output_raw_new.append(sys)

            #output_raw_delex_new.append('<|response|> ' + sys_delex.strip() + ' <|endofresponse|>')
            output_raw_delex_new.append('assistant\n\n'+sys_delex.strip())

            db_text = dbPointer.convert_dbpointer_to_text(db_data[i], goal, d_lex['belief_raw'][i])
            #db_search_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text))
            db_search_raw.append(db_text)

            db_text_nmatch = dbPointer.convert_dbpointer_to_text_nmatch(db_data[i], goal, d_lex['belief_raw'][i])
            #db_nmatch_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text_nmatch))
            db_nmatch_raw.append(db_text_nmatch)

        belief = d_lex['belief_raw']
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if b[-1] in ['not mentioned']: # comment this for DST task
                    continue
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            #tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            tmp_new = ' , '.join(tmp_bs_new)
            belief_raw_new.append(tmp_new)

        # belief for DST task (include none)
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            #tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            tmp_new = ' , '.join(tmp_bs_new)
            belief_raw_none_new.append(tmp_new)

        action = d_lex['action_raw']
        for act in action:
            tmp_act_new = []
            for i, a in enumerate(act):
                if i == len(act) - 1:
                    tmp_act_new.append(' '.join(a))
                else:
                    tmp_act_new.append(' '.join(a))
            if len(tmp_act_new) == 0:
                tmp_act_new.append(' ')

            #tmp_new = '<|action|> {} <|endofaction|>'.format(' , '.join(tmp_act_new))
            tmp_new = ' , '.join(tmp_act_new)
            action_raw_new.append(tmp_new)

    def save_dial_msg(history=False,
                        belief=False,
                        dbsearch=False,
                        dbnmatch=False,
                        action=False,
                        response=False):
        tmp = []
        
        for inp, bs, dbs, dbnm, act, trg in zip(history_raw_new,
                                                belief_raw_new,
                                                db_search_raw,
                                                db_nmatch_raw,
                                                action_raw_new,
                                                output_raw_delex_new):
            messages = []
            if history: messages.append({"role": "context", "content": inp.lower()})
            if belief: messages.append({"role": "belief", "content": bs.lower()})
            if dbsearch: messages.append({"role": "dbsearch", "content": dbs.lower()})
            if dbnmatch: messages.append({"role": "dbsearch", "content": dbnm.lower()})
            if action: messages.append({"role": "action", "content": act.lower()})
            if response: messages.append({"role": "response", "content": trg.lower()})
            tmp.append(llama3_tknzer.decode(llama3_tknzer.apply_chat_template(messages)))
        file_path = '{}/{}.'.format(save_dir, split)
        suffix = []
        if history: suffix += ['history']
        if belief: suffix += ['belief']
        if dbsearch: suffix += ["dbsearch"]
        if dbnmatch: suffix += ['dbnmatch']
        if action: suffix += ['action']
        if response: suffix += ['sys_delex']

        file_path += '_'.join(suffix)
        with open(file_path, 'wt') as f:
            for l in tmp:
                f.write('{}\n'.format(l))
        with open(file_path+'.pkl', "wb") as f:
            pickle.dump(tmp, f)

    save_dial_msg(history=True, belief=True, dbsearch=True, action=True, response=True)
    save_dial_msg(history=True, belief=True, dbnmatch=True, action=True, response=True)
    save_dial_msg(history=True)
    save_dial_msg(history=True, belief=True)
    save_dial_msg(history=True, belief=True, action=True, response=True)
    
