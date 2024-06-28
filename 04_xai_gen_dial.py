
from typing import List, Tuple

import torch
import tokenizers
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
from evaluate_multiwoz import MultiWozDB
from utils.multiwoz import dbPointer
from utils.simpletod import *
import tqdm
import json
import ipdb
import sys
import os
import re



opt = ArgsParser().parse()
opt.multiwoz_version = '2.1'
opt.use_action = True
opt.use_knowledge = True
opt.context_knowledge = True
opt.lexical = True



HISTORY_LEN = None
# USE_ORACLE_BELIEF = True
USE_ORACLE_BELIEF = opt.use_oracle_belief
# USE_ORACLE_ACTION = False
USE_ORACLE_ACTION = opt.use_oracle_action
# USE_DB_SEARCH = True
USE_DB_SEARCH = opt.use_db_search
USE_DYNAMIC_DB = opt.use_dynamic_db
# EVAL_SPLIT = 'test'
EVAL_SPLIT = opt.split_set
FASTGEN_OUTPUT_LEN = 192

decoding = opt.decoding

multiwoz_data = json.load(open('resources/multi-woz/lex.json','r'))

# provide model name and checkpoint directory
# exp_name = 'gpt2'
# exp_name = opt.experiment_name
# checkpoint = opt.checkpoint
# model_checkpoint = '../dialog-transformer/output/{}/{}/'.format(exp_name, checkpoint)
model_checkpoint = opt.checkpoint
exp_name = os.path.split(model_checkpoint)[0].split('/')[-2]
ckpt_num = model_checkpoint.split('-')[-1]
exp_name += ckpt_num
print(exp_name)


multiwoz_db = MultiWozDB()

opt_delex = ArgsParser().parse()
opt_delex.multiwoz_version = '2.1'


data = MultiWozDataset(opt, split=EVAL_SPLIT, shuffle=False)

data_delex = MultiWozDataset(opt_delex, split=EVAL_SPLIT, shuffle=False)

lex_dict = {}
delex_dict = {}
max_num = 10
for d in data:
    lex_dict[d['name']] = d
    if len(lex_dict) == max_num: break

for d in data_delex:
    delex_dict[d['name']] = d
    if len(delex_dict) == max_num: break

if 'openai-gpt' in model_checkpoint:
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
elif "llama" in model_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

if 'openai-gpt' in model_checkpoint:
    model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
elif "llama" in model_checkpoint:
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                 torch_dtype = torch.bfloat16,
                                                 device_map="cuda:0")
else:
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

model.eval()
#model.to('cuda')

break_tokens = tokenizer.encode("{}".format(tokenizer._eos_token), add_special_tokens=False)
# TODO: load max input length properly
MAX_LEN = 1024 
#MAX_LEN = model.config.n_ctx 


generated_dict = {}
num_data = len(data)

hbegin_str = "<|start_header_id|>"
hend_str = "<|end_header_id|>"
get_header_str = lambda head_id: f"{hbegin_str}{head_id}{hend_str}"
eot_str = tokenizer.eos_token

def get_first_appear(header_id, text, end_with_other_header=True) -> Tuple[str, Tuple[int, int]]:
    """
    param:
        - header_id:
            찾아서 내용을 파싱하고자 하는 헤더 문자열
        - text:
            원본 텍스트, 헤더 찾는 곳
        - end_with_other_header:
          eot 토큰으로 끝나지 않아도 다른 헤더로 끊음
          자연어 모델이 eot 토큰을 생성하지 않는 일부 경우에 대응함.
    returns:
        - parse_text:
            파싱된 텍스트. 찾지 못했다면 빈 문자열 반환
        - (begin, end)
            원본 텍스트 (text)에서의 파싱된 텍스트의 위치. 찾지 못했다면 (-1, -1) 반환
    """
    header_str = get_header_str(header_id)
    header_begin = text.find(header_str)
    if header_begin == -1:
        return "", (-1, -1)

    eot_begin = text[header_begin:].find(eot_str)
    if end_with_other_header:
        other_header_begin = text[header_begin+len(hbegin_str):].find(hbegin_str)
        if other_header_begin != -1:
            other_header_begin += len(hbegin_str) #eot_begin와의 offset 반영
            if eot_begin == -1:
                eot_begin = other_header_begin - len(eot_str)
            else:
                eot_begin = min(eot_begin , other_header_begin-len(eot_str))
    
    if eot_begin == -1:    
        eot_begin = len(text)-len(eot_str)
    else:
        eot_begin += header_begin
    parsed_text = text[header_begin:eot_begin+len(eot_str)]
    return parsed_text, (header_begin, eot_begin+len(eot_str))
def remove_header_and_eot(text) -> str:
    """
    주어진 text에서 맨 앞에 등장한 header와 맨 뒤의 eot를 제거하고 그 사이 텍스트를 반환합니다.
    중복된 eot와 header가 등장할 시 제거되지 않는 등 오작동할 수 있습니다.
    eot가 없는 경우에는 헤더만 제거합니다.
    추후 앞 뒤 개행문자와 공백을 제거합니다.
    """
    text = text.split(hend_str,maxsplit=1)[-1]
    if text.find(eot_str) != -1:
        text = eot_str.join(text.split(eot_str)[:-1])
    text = text.strip("\n ")
    return text



if USE_DB_SEARCH and USE_ORACLE_BELIEF:
    with open('resources/llama3/test.history_belief_dbsearch_action_sys_delex_key.json','r') as f:
        dial_desired_outputs =json.load(f)
else:     
    with open('resources/llama3/test.history_belief_action_sys_delex_key.json','r') as f:
        dial_desired_outputs =json.load(f)


for i, dial_name in enumerate(lex_dict):
    if EVAL_SPLIT == 'train' and i > 1000:
        break
    d = lex_dict[dial_name]
    d_delex = delex_dict[dial_name]
    print('{} [{}/{}]\n'.format(d['name'], i, num_data), end='')
    sys.stdout.flush()
    beliefs_raw = d['belief_raw']
    user = d['input_raw']
    system = d['target_raw']
    system_delex = d_delex['target_raw']
    if 'delex' in model_checkpoint:
        target_response = system_delex
    else:
        target_response = system


    action = d['action_raw']
    target_action = []
    for turn_act in action:
        turn_action = []
        for act in turn_act:
            act_str = '{} {} {}'.format(act[0], act[1], act[2])
            turn_action.append(act_str)
        target_action.append(turn_action)

    dialogue_aggregated_target_belief = []
    dialogue_target_belief = []

    for turn_belief in beliefs_raw:
        turn_belief_str = []
        for bs in turn_belief:
            domain, slot, value = bs
            if value in ['not mentioned', 'none']:
                continue
            bs_str = '{} {} {}'.format(domain.lower(), slot.lower(), value.lower())
            if bs_str not in dialogue_aggregated_target_belief:
                dialogue_aggregated_target_belief.append(bs_str)
            turn_belief_str.append(bs_str)
        dialogue_target_belief.append(turn_belief_str)


    db_data = d['db']
    goal = multiwoz_data[dial_name]['goal']

    dial_desireds = dial_desired_outputs[dial_name]

    generated_raw = []
    generated = []
    model_context = []
    for turn_id, (usr_turn, _) in enumerate(zip(user, system)):
        tmp_text = dial_desireds[turn_id]
        text = re.sub(r"\\n", r"\n", tmp_text)
        if dial_name == 'SNG02319.json':
            text = text.replace('300 will', '03:00 will')
        
        if USE_DB_SEARCH:
            db_header = get_header_str("dbsearch")
            db_text = f"{db_header}{text.split(db_header)[1].split(eot_str)[0]}{eot_str}"
        
        header_id = "response"
        if not USE_ORACLE_BELIEF:
            header_id = "belief"
        elif not USE_ORACLE_ACTION:
            header_id = "action"
        else:
            pass # header_id = response
        header = get_header_str(header_id)+"\n\n"
        header_begin = text.find(header)
        text = text[:header_begin+len(header)]


        """
        # TODO temp fix. i don't know why it happened but they started with `<|begin_of_text|><|begin_of_text|>`
        #text = '{} <|context|> {} <|endofcontext|> '.format(tokenizer._bos_token, tmp_text)
        text = ' <|context|> {} <|endofcontext|> '.format(tmp_text)


        if USE_ORACLE_BELIEF:
            turn_belief = dialogue_target_belief[turn_id]
            belief_str = '<|belief|> {} <|endofbelief|>'.format(' , '.join(turn_belief))
            text = text + ' ' + belief_str
        
        db_text = dbPointer.convert_dbpointer_to_text(db_data[turn_id], goal, beliefs_raw[turn_id])
        if USE_DB_SEARCH and USE_ORACLE_BELIEF:
            if not USE_ORACLE_BELIEF:
                print('warning: oracle db is true, oracle belief is false')
            text += ' <|dbsearch|> {} <|endofdbsearch|>'.format(db_text)
        
        if USE_ORACLE_ACTION:
            turn_action = target_action[turn_id]
            action_str = '<|action|> {} <|endofaction|>'.format(' , '.join(turn_action))
            text = text + ' ' + action_str
        """
        #print(text)
        #print("=====")
        model_context.append(text)
        indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1*MAX_LEN:]

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        if USE_DB_SEARCH and not USE_ORACLE_BELIEF: # generate belief, then get DB search results, then continue generation (greedy decoding)
            # TODO: well... actually... no
            assert False
            with torch.no_grad():
                while predicted_index not in break_tokens:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]
                    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                    if len(indexed_tokens) > MAX_LEN:
                        break
                    if tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                        break

            tmp_pred = tokenizer.decode(indexed_tokens)
            if not USE_DYNAMIC_DB: # use oracle db
                text = '{} {}'.format(tmp_pred, db_text)
            else: # use dynamic db search results (using generated belief)
                db_text_dynamic = get_db_dynamically(tmp_pred, goal, multiwoz_db=multiwoz_db)
                text = '{} {}'.format(tmp_pred, db_text_dynamic)

            # continue generation
            indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(indexed_tokens) > MAX_LEN:
                indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            predicted_index = indexed_tokens[-1]

            # Predict all tokens
            with torch.no_grad():
                #while predicted_index not in break_tokens:
                truncate_action = False
                while predicted_index not in break_tokens:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]

                    # sometime model generate repeated actions, we just use truncate actions if this happens
                    predicted_text = tokenizer.decode(indexed_tokens)
                    if '<|action|>' in predicted_text:
                        generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[
                            0].split(',')
                        new_actions = []
                        for a in generated_actions:
                            if a in ['', ' ']:
                                continue
                            new_actions.append(a.strip())
                        len_actions = len(new_actions)
                        if len(list(set(new_actions))) > len(new_actions) or (
                                len_actions > 10 and not truncate_action):
                            actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                            indexed_tokens = tokenizer.encode(
                                '{} {}'.format(predicted_text.split('<|action|>')[0], actions), add_special_tokens=False,)
                            truncate_action = True

                    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                    if len(indexed_tokens) > MAX_LEN:
                        break
                    if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                        break

                predicted_text = tokenizer.decode(indexed_tokens)
                generated.append(predicted_text)

        else: # generate belief, action, and response once
            with torch.no_grad():

                if decoding == 'nucleus':
                    sample_output = model.generate(
                        tokens_tensor,
                        # indexed_tokens,
                        do_sample=True,
                        max_length=min(MAX_LEN, len(tokens_tensor[0])+FASTGEN_OUTPUT_LEN) if FASTGEN_OUTPUT_LEN and FASTGEN_OUTPUT_LEN > 0 else MAX_LEN,
                        top_p=0.5,
                        top_k=0,
                        pad_token_id = tokenizer.eos_token_id,
                        eos_token_id = [],
                    )
                    predicted_text_raw = tokenizer.decode(sample_output[0])
                    #print(predicted_text_raw)
                    #print("======\n======")
                    response_header = get_header_str("response")
                    response_begin = predicted_text_raw.find(response_header)
                    if response_begin == -1:
                        tmp = predicted_text_raw.split(eot_str)
                        tmp = [t.strip() for t in tmp]
                        predicted_text = eot_str.join([t for t in tmp if t != ""])
                    else:
                        resp_eot_begin = predicted_text_raw[response_begin:].find(eot_str)
                        if resp_eot_begin == -1:
                            resp_eot_begin = len(predicted_text_raw)-len(eot_str)
                        else:
                            resp_eot_begin += response_begin
                        predicted_text = predicted_text_raw[:resp_eot_begin+len(eot_str)]
                    #tmp = ' '.join([predicted_text_raw.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    #predicted_text = tmp
                    generated_raw.append(predicted_text_raw)
                    generated.append(predicted_text)
                    del predicted_text_raw
                    del predicted_text

                elif decoding == 'greedy':
                    assert False
                    # GREEDY DECODING
                    
                    # sample_output = model.generate(
                    #     # tokens_tensor,
                    #     indexed_tokens,
                    #     max_length=MAX_LEN,
                    #     do_sample=False
                    # )
                    truncate_action = False
                    while predicted_index not in break_tokens:
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
                        indexed_tokens += [predicted_index]
                        
                        # sometime model generate repeated actions, we just use truncate actions if this happens
                        predicted_text = tokenizer.decode(indexed_tokens)
                        if '<|action|>' in predicted_text:
                            generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[
                                0].split(',')
                            new_actions = []
                            for a in generated_actions:
                                if a in ['', ' ']:
                                    continue
                                new_actions.append(a.strip())
                            len_actions = len(new_actions)
                            if len(list(set(new_actions))) > len(new_actions) or (
                                    len_actions > 10 and not truncate_action):
                                actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                                indexed_tokens = tokenizer.encode(
                                    '{} {}'.format(predicted_text.split('<|action|>')[0], actions), add_special_tokens=False,)
                                truncate_action = True
                                
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                            break
                            
                    predicted_text = tokenizer.decode(indexed_tokens)
                    tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    generated.append(predicted_text)

                # predicted_text = tokenizer.decode(sample_output[0])
                # tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                # predicted_text = tmp
                # generated.append(predicted_text)

    # TODO
    #generated_dict[d['name']] = {
    #    'target_belief': dialogue_aggregated_target_belief,
    #    'target_turn_belief': dialogue_target_belief,
    #    'target_response': target_response,
    #    'generated': generated,
    #    'generated_raw': generated_raw,
    #    'target_action': target_action,
    #    'target_user': user,
    #    'model_context': model_context
    #}
    #
    dialogue_aggregated_pred_belief = []
    dialogue_pred_belief = []
    dialogue_pred_responses = []
    dialogue_pred_action = []
    # aggregate belief states
    for turn, pred in enumerate(generated):
        # 믿음 상태 얻기
        parsed_belief, b_e = get_first_appear("belief", pred)
        new_belief = set()
        if b_e[0] != -1:
            parsed_belief = remove_header_and_eot(parsed_belief)
            belief = parsed_belief.split(',')
            for bs in belief:
                bs = bs.strip(' .,\n')
                if bs == '': continue
                if bs not in new_belief:
                    new_belief.add(bs)
        new_belief=list(new_belief)
        if len(new_belief) == 0:
            new_belief = ['']
            # 이번 차례에 생성된 belief가 없는 경우에 이전 차례 값으로 '봐주기'
            # belief 특혜?
            if len(dialogue_pred_belief) > 0:
                new_belief = dialogue_pred_belief[-1]
        dialogue_pred_belief.append(new_belief)
        dialogue_aggregated_pred_belief += [bs for bs in new_belief if bs not in ['', ' ']+dialogue_aggregated_pred_belief]
        

        # 대화 행동 얻기
        parsed_action, b_e = get_first_appear("action", pred)
        new_action = set()
        if b_e[0] != -1:
            parsed_action = remove_header_and_eot(parsed_action)
            action = parsed_action.split(',')
            for act in action:
                act = act.strip(' .,\n')
                if act == '': continue
                if act not in new_action:
                    new_action.add(act)
        new_action=list(new_action)
        # belief랑 다르게 빈 str 넣어주는게 없네?
        #if len(new_action) == 0:
        #    new_action = ['']
        dialogue_pred_action.append(new_action)


        # 응답 얻기
        parsed_resp, b_e = get_first_appear("response", pred)
        if b_e[0] != -1:
            parsed_resp = remove_header_and_eot(parsed_resp)
            #assistant\n\n 가 있다면 발견해서 지웁니다.
            parsed_resp = parsed_resp.replace("assistant\n\n","")
            parsed_resp = parsed_resp.strip(' .,\n')
        else:
            parsed_resp = ''
        dialogue_pred_responses.append(parsed_resp)

    generated_dict[d['name']] = {
        'target_belief': dialogue_aggregated_target_belief,
        'target_turn_belief': dialogue_target_belief,
        'generated_belief': dialogue_aggregated_pred_belief,
        'generated_turn_belief': dialogue_pred_belief,
        'target_response': target_response,
        'generated_response': dialogue_pred_responses,
        'target_action': target_action,
        'generated_action': dialogue_pred_action,
        'target_user': user,
        'model_context': model_context,
        'generated': generated,
        'generated_raw': generated_raw,
    }


save_name = '{}_{}'.format(exp_name, EVAL_SPLIT)

if USE_ORACLE_BELIEF:
    save_name += '_oracleBelief'

if USE_DB_SEARCH:
    save_name += '_oracleDB'

if USE_ORACLE_ACTION:
    save_name += '_oracleAction'

if HISTORY_LEN:
    save_name += '_context[history={}]'.format(HISTORY_LEN)
else:
    save_name += '_context[history=full_history]'

save_name += '_nocarry'

with open('{}.json'.format(save_name), 'wt') as f:
    json.dump(generated_dict, f)
