
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

os.environ["TOKENIZERS_PARALLELISM"]="false"

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
USE_MULTIPROCESSING = True

decoding = opt.decoding

multiwoz_data = json.load(open('resources/multi-woz/lex.json','r'))

# provide model name and checkpoint directory
# exp_name = 'gpt2'
# exp_name = opt.experiment_name
# checkpoint = opt.checkpoint
# model_checkpoint = '../dialog-transformer/output/{}/{}/'.format(exp_name, checkpoint)
model_checkpoint = opt.checkpoint
exp_name = "/".join(os.path.split(model_checkpoint)[0].split('/')[:-1])
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
max_num = 100
for d in data:
    lex_dict[d['name']] = d
    if len(lex_dict) == max_num: break

for d in data_delex:
    delex_dict[d['name']] = d
    if len(delex_dict) == max_num: break

if 'openai-gpt' in model_checkpoint:
    tknzer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
    tknzer.add_special_tokens({'bos_token': '<|endoftext|>'})
    tknzer.add_special_tokens({'eos_token': '<|endoftext|>'})
elif "llama" in model_checkpoint:
    tknzer = AutoTokenizer.from_pretrained(model_checkpoint)
    tknzer.pad_token_id = tknzer.eos_token_id
else:
    tknzer = GPT2Tokenizer.from_pretrained(model_checkpoint)

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

break_tokens = tknzer.encode("{}".format(tknzer._eos_token), add_special_tokens=False)
# TODO: load max input length properly
MAX_LEN = 1024 
#MAX_LEN = model.config.n_ctx 


generated_dict = {}
num_data = len(data)

hbegin_str = "<|start_header_id|>"
hend_str = "<|end_header_id|>"
get_header_str = lambda head_id: f"{hbegin_str}{head_id}{hend_str}"
EOT_STR = "<|eot_id|>"

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

    eot_begin = text[header_begin:].find(EOT_STR)
    if end_with_other_header:
        other_header_begin = text[header_begin+len(hbegin_str):].find(hbegin_str)
        if other_header_begin != -1:
            other_header_begin += len(hbegin_str) #eot_begin와의 offset 반영
            if eot_begin == -1:
                eot_begin = other_header_begin - len(EOT_STR)
            else:
                eot_begin = min(eot_begin , other_header_begin-len(EOT_STR))
    
    if eot_begin == -1:    
        eot_begin = len(text)-len(EOT_STR)
    else:
        eot_begin += header_begin
    parsed_text = text[header_begin:eot_begin+len(EOT_STR)]
    return parsed_text, (header_begin, eot_begin+len(EOT_STR))
def remove_header_and_eot(text) -> str:
    """
    주어진 text에서 맨 앞에 등장한 header와 맨 뒤의 eot를 제거하고 그 사이 텍스트를 반환합니다.
    중복된 eot와 header가 등장할 시 제거되지 않는 등 오작동할 수 있습니다.
    eot가 없는 경우에는 헤더만 제거합니다.
    추후 앞 뒤 개행문자와 공백을 제거합니다.
    """
    text = text.split(hend_str,maxsplit=1)[-1]
    if text.find(EOT_STR) != -1:
        text = EOT_STR.join(text.split(EOT_STR)[:-1])
    text = text.strip("\n ")
    return text

def get_belief_dbsearch_llama(sent):
    """
    나만의 버전
    믿음 상태 헤더의 내용물을 파싱하고 문자열로 나열된 믿음 상태를 리스트로 정리한다.
    """
    parsed, b_e = get_first_appear("belief", sent)
    if b_e[0] != -1:
        tmp = remove_header_and_eot(parsed)
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('<|start_header_id|>', '')
    tmp = tmp.replace('<|end_header_id|>', '')
    tmp = tmp.replace(EOT_STR, '')
    tmp = tmp.replace("<|end_of_text|>", '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_db_dynamically_llama(predicted_text, goal, multiwoz_db):
    """
    수정점: 믿음 상태를 split 했을 때 일정 길이 이상이 되지 않는다면 continue 하도록 변경하였습니다.
    그렇게 하지 않는 경우 짧게 출력해버린 경우에 오류를 내기 때문입니다.
    """
    gen_belief = get_belief_dbsearch_llama(predicted_text)
    belief_domain = {}
    belief_book_domain = {}
    for bs in gen_belief:
        if bs in ['', ' ']:
            continue
        bs_split = bs.split()
        bs_domain = bs_split[0]
        if 'book' in bs:
            if len(bs_split) < 3: continue
            bs_slot = bs_split[2]
            bs_val = ' '.join(bs_split[3:])
            if bs_domain not in belief_book_domain:
                belief_book_domain[bs_domain] = {}
            belief_book_domain[bs_domain][bs_slot] = bs_val
        else:
            if len(bs_split) < 2: continue
            bs_slot = bs_split[1]
            bs_val = ' '.join(bs_split[2:])
            if bs_domain not in belief_domain:
                belief_domain[bs_domain] = {}
                belief_book_domain[bs_domain] = {}
            belief_domain[bs_domain][bs_slot] = bs_val

    db_text_tmp = []
    for dom in belief_domain:
        if dom not in ['restaurant', 'hotel', 'attraction', 'train']:
            continue
        domain_match = len(multiwoz_db.queryResultVenues(dom, belief_domain[dom], real_belief=True))

        if dom != 'train':
            if domain_match >= 5:
                domain_match_text = '>=5'
            else:
                domain_match_text = '={}'.format(domain_match)

        elif dom == 'train':
            if domain_match == 0:
                domain_match_text = '=0'
            elif domain_match == 2:
                domain_match_text = '<3'
            elif domain_match == 5:
                domain_match_text = '<6'
            elif domain_match == 10:
                domain_match_text = '<11'
            elif domain_match == 40:
                domain_match_text = '<41'
            else:
                domain_match_text = '>40'
        #it also can be fix by
        #if 'fail_book' in goal[dom]:
        #    domain_book_text = 'available'
        if 'fail_book' in goal[dom] and len(goal[dom]['fail_book']) > 0:
            for item in goal[dom]['fail_book'].items():
                if item in belief_book_domain[dom].items():
                    domain_book_text = 'not available'
                    break
                else:
                    domain_book_text = 'available'
        else:
            if domain_match == 0:
                domain_book_text = 'not available'
            else:
                domain_book_text = 'available'

        db_text_tmp.append('{} match{} booking={}'.format(dom, domain_match_text, domain_book_text))
    db_text = '{}\n\n{}{}'.format(
                                get_header_str("dbsearch"),
                                ' , '.join(db_text_tmp),
                                EOT_STR
                            )
    return db_text

def text2token(text, tokenizer, maxlen=-1):
    """주어진 텍스트를 주어진 토크나이저로 토큰화하고 텐서로 전환하여 전달합니다."""
    indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
    if maxlen != -1 and len(indexed_tokens) > maxlen:
        indexed_tokens = indexed_tokens[-1*maxlen:]
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    return tokens_tensor

    
def gen_dial(model,
             tokenizer,
             tokens_tensor,
             decoding = "nucleus",
             last_header_id = "response",
             forced_eot_end=False
             ):
    """
    토큰화된 텐서를 입력으로 모델에게 생성하도록 요청합니다.
    해당하는 header의 내용이 마지막 출력이 되도록 합니다.
    
    params:
    - model:
        언어모델입니다. model.generate()를 호출합니다.
    - tokenizer:
        사용하는 토크나이저입니다. eot_token_id 정보와 decode()를 위해 사용합니다.
    - tokens_tensor:
        토큰 id로 이루어진 torch 텐서입니다.
    - decoding:
        디코딩 종류입니다.
        TODO: greedy를 지원하도록 코드 추가
    - last_header_id:
        해당 header가 출력되면 그 내용까지만을 리턴으로 반환합니다.
        해당 header가 출력되지 않았거나 eot 토큰이 나오지 않았으면 끝까지 출력합니다.
        만약 ""이나 None, False라면, 무조건 끝까지 출력합니다.
        주의: 입력 포함된 header를 인자로 주면 최종 반환 텍스트가 생성된 내용을 전부 자를 수 있습니다.
    - forced_eot_end:
        출력 끝에 eot가 나오지 않았다면 eot 토큰을 강제로 붙입니다.
    
    returns:
    - predicted_text:
        출력 된 후 파싱된 텍스트.
    - predicted_text_raw:
        전처리를 거치지 않은 모델의 출력 텍스트

    """
    if decoding == "nucleus":
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
    else:
        raise NotImplementedError(f"Not supported decoding type: {decoding}")
    predicted_text_raw = tokenizer.decode(sample_output[0])

    if last_header_id:
        header_str = get_header_str(last_header_id)
        header_begin = predicted_text_raw.find(header_str)
        if header_begin == -1:
            tmp = predicted_text_raw.split(EOT_STR)
            tmp = [t.strip() for t in tmp]
            predicted_text = EOT_STR.join([t for t in tmp if t != ""])
        else:
            eot_begin = predicted_text_raw[header_begin:].find(EOT_STR)
            if eot_begin == -1:
                eot_begin = len(predicted_text_raw)-len(EOT_STR)
            else:
                eot_begin += header_begin
            predicted_text = predicted_text_raw[:eot_begin+len(EOT_STR)]
    if forced_eot_end and (not predicted_text.endswith(EOT_STR)):
        predicted_text += EOT_STR
    return predicted_text, predicted_text_raw



if USE_DB_SEARCH:
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
    tgt_raw = []
    for turn_id, (usr_turn, _) in enumerate(zip(user, system)):
        tmp_text = dial_desireds[turn_id]
        text = re.sub(r"\\n", r"\n", tmp_text)
        if dial_name == 'SNG02319.json':
            text = text.replace('300 will', '03:00 will')
        tgt_raw.append(text)
        
        if USE_DB_SEARCH and (not USE_DYNAMIC_DB):
            db_header = get_header_str("dbsearch")
            db_text = f"{db_header}{text.split(db_header)[1].split(EOT_STR)[0]}{EOT_STR}"
        
        first_tgt_header_id = "response"
        if not USE_ORACLE_BELIEF:
            first_tgt_header_id = "belief"
        elif not USE_ORACLE_ACTION:
            first_tgt_header_id = "action"
        else:
            pass # first_tgt_header_id = response
        first_tgt_header = get_header_str(first_tgt_header_id)+"\n\n"
        first_tgt_header_begin = text.find(first_tgt_header)
        text = text[:first_tgt_header_begin+len(first_tgt_header)]

        model_context.append(text)
        tokens_tensor = text2token(text, tknzer, MAX_LEN)
        predicted_index = tokens_tensor[0][-1].item()

        if USE_DB_SEARCH and not USE_ORACLE_BELIEF: # generate belief, then get DB search results, then continue generation (greedy decoding)
            if decoding == 'nucleus':
                # 생성 후 믿음 상태까지만 가져오기 
                predicted_text, _ = gen_dial(
                    model,
                    tknzer,
                    tokens_tensor,
                    decoding="nucleus",
                    last_header_id="belief",
                    forced_eot_end=True,
                )
                #tmp = ' '.join([predicted_text_raw.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                #predicted_text = tmp
                
                # 믿음 상태 다음에 DB 검색 결과 붙이기
                if not USE_DYNAMIC_DB:
                    text_w_db = f"{predicted_text}{db_text}"
                else:
                    db_text_dynamic = get_db_dynamically_llama(predicted_text, goal, multiwoz_db=multiwoz_db)
                    text_w_db = f"{predicted_text}{db_text_dynamic}"
                

                # 믿음+DB 결과로 행동과 응답 한번에 생성
                tokens_tensor = text2token(text_w_db, tknzer, MAX_LEN)
                predicted_index = tokens_tensor[0][-1].item()

                predicted_text, predicted_text_raw = gen_dial(
                    model,
                    tknzer,
                    tokens_tensor,
                    decoding="nucleus",
                    last_header_id="response",
                    forced_eot_end=False,
                )
                generated_raw.append(predicted_text_raw)
                generated.append(predicted_text)

            elif decoding == 'greedy':
                assert False

        else: # generate belief, action, and response once
            with torch.no_grad():

                if decoding == 'nucleus':
                    predicted_text, predicted_text_raw = gen_dial(
                        model,
                        tknzer,
                        tokens_tensor,
                        decoding="nucleus",
                        last_header_id="response",
                        forced_eot_end=False
                        )
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
                        predicted_text = tknzer.decode(indexed_tokens)
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
                                indexed_tokens = tknzer.encode(
                                    '{} {}'.format(predicted_text.split('<|action|>')[0], actions), add_special_tokens=False,)
                                truncate_action = True
                                
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        if tknzer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                            break
                            
                    predicted_text = tknzer.decode(indexed_tokens)
                    tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    generated.append(predicted_text)

                # predicted_text = tknzer.decode(sample_output[0])
                # tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                # predicted_text = tmp
                # generated.append(predicted_text)
    if USE_MULTIPROCESSING:
        generated_dict[d['name']] = {
            'target_belief': dialogue_aggregated_target_belief,
            'target_turn_belief': dialogue_target_belief,
            'target_response': target_response,
            'generated': generated,
            'generated_raw': generated_raw,
            'target_raw': tgt_raw,
            'target_action': target_action,
            'target_user': user,
            'model_context': model_context
        }
    else:
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
            'target_raw': tgt_raw,
        }

if USE_MULTIPROCESSING:
    print("start parse and saving...(multiprocessing)")
    import multiprocessing as mp

    def mp_parsing(generated):
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
        return (
            dialogue_aggregated_pred_belief,
            dialogue_pred_belief,
            dialogue_pred_responses,
            dialogue_pred_action
        )

    with mp.Pool(14) as pool:
        generateds = [generated_dict[key]["generated"] for key in generated_dict.keys()]
        for ret, key in zip(pool.imap(mp_parsing, generateds), generated_dict.keys()):
            dialogue_aggregated_pred_belief,\
                dialogue_pred_belief,\
                dialogue_pred_responses,\
                dialogue_pred_action = ret
            generated_dict[key]["generated_belief"] = dialogue_aggregated_pred_belief
            generated_dict[key]["generated_turn_belief"] = dialogue_pred_belief
            generated_dict[key]["generated_response"] = dialogue_pred_responses
            generated_dict[key]["generated_action"] = dialogue_pred_action

        

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
