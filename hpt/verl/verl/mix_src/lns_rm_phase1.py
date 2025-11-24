import torch
import numpy as np
import os
import re
import json
from verl import DataProto

# =============================================================================
# 1. Utility Functions (Metric & Format)
# =============================================================================

def simple_format_checker(solution_str):
    """
    Assistant의 전체 대화 기록(solution_str)을 검사하여,
    모든 턴이 정해진 문법 규칙을 따랐는지 확인합니다.
    """
    # 1단계: Assistant의 턴(Turn)만 분리하기
    assistant_turns = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.DOTALL)

    if not assistant_turns:
        return 0.0

    # 2단계: 모든 턴에 대한 '공통 문법' 검사
    for i, turn in enumerate(assistant_turns):
        cleaned_turn = turn.strip()

        # 2-A: <think> 태그 검사
        if not cleaned_turn.startswith('<think>'):
            return 0.0
        if cleaned_turn.count('<think>') != 1 or cleaned_turn.count('</think>') != 1:
            return 0.0

        # 2-B: 행동(Action) 태그 개수 검사
        action_count = cleaned_turn.count('<search>') + cleaned_turn.count('<bbox>') + cleaned_turn.count('<search_complete>')
        if action_count != 1:
            return 0.0

        # 2-C: 행동(Action) 태그 내용 검사 (세부 규칙)
        if '<search>' in cleaned_turn:
            match = re.search(r"<search>(.*?)</search>", cleaned_turn, re.DOTALL)
            if not match or not match.group(1).strip():
                return 0.0

        elif '<bbox>' in cleaned_turn:
            match = re.search(r"<bbox>(.*?)</bbox>", cleaned_turn, re.DOTALL)
            if not match:
                return 0.0
            try:
                bbox_content = json.loads(match.group(1).strip())
                if not isinstance(bbox_content, list) or len(bbox_content) != 4:
                    return 0.0
                if not all(isinstance(coord, (int, float)) for coord in bbox_content):
                    return 0.0
            except json.JSONDecodeError:
                return 0.0

        elif '<search_complete>' in cleaned_turn:
            # 공백 제거 후 확인
            if '<search_complete>true</search_complete>' not in cleaned_turn.replace(" ", ""):
                return 0.0

    # 3단계: '마지막 턴' 특별 규칙 검사
    last_turn = assistant_turns[-1].strip()
    if '<search_complete>' not in last_turn:
        return 0.0

    # 4단계: 최종 합격 판정
    return 1.0

def dcg(relevance_scores):
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    dcg_value = dcg(relevance_scores)
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value

def recall_ret(sorted_docs, golden_answer_list):
    sorted_docs_set = set(sorted_docs)
    golden_answer_set = set(golden_answer_list)
    if len(golden_answer_set) == 0:
        return 0.0
    return len(sorted_docs_set.intersection(golden_answer_set)) / len(golden_answer_set)


# =============================================================================
# 2. Training Reward Manager (for mix_trainer.py)
# =============================================================================

class RagRewardManagerPhase1:
    def __init__(self, tokenizer, num_examine):
        # rm_url 등 외부 평가 관련 인자 제거
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        """학습용: 최종 Reward Tensor 하나만 반환"""
        print("리워드 함수 작동")

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            # Prompt 부분 마스킹 계산
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            # Response 부분 추출
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 텍스트 디코딩
            full_text = self.tokenizer.decode(torch.cat([valid_prompt_ids, valid_response_ids]))

            # 1. Format Reward 계산
            format_score = simple_format_checker(full_text)
            
            # Format을 통과한 경우에만 nDCG 계산 및 점수 부여
            if format_score > 0.0:
                ndcg_val = 0.0
                try:
                    # 배치 데이터에서 검색된 이미지 및 정답 정보 가져오기
                    #extra_info = data_item.non_tensor_batch.get('extra_info', {})
                    retrieved_imgs = data_item.non_tensor_batch.get('retrievaled_images', [])
                    
                    retrievaled_base = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in retrieved_imgs]
                    
                    #ref_pages = extra_info.get('reference_page', [])
                    if isinstance(ref_pages, np.ndarray): ref_pages = ref_pages.tolist()
                    file_name = extra_info.get('file_name', 'unknown.pdf')
                    reference_base = [f'{file_name.split(".pdf")[0]}_{p}' for p in ref_pages]
                    
                    ndcg_val = ndcg(retrievaled_base, reference_base)
                except Exception:
                    ndcg_val = 0.0

                # 2. 최종 점수 계산 (수정된 가중치 적용)
                # Weight: Format: 0.1, nDCG: 0.9
                final_score = (0.1 * 1.0) + (0.9 * ndcg_val)

                # ========== [디버깅 출력 시작] ==========
                print("===================== RAG REWARD (TRAIN) DEBUG =====================")
                print(f"[Sample {i}] Format PASS (Score 1.0)")
                print(f"  - Model Retrieved Images (Base): {retrievaled_base}")
                print(f"  - Reference Images (Base):       {reference_base}")
                print(f"  - Calculated nDCG Value:         {ndcg_val:.4f}")
                print(f"  - FINAL SCORE CALC:              (0.1 * 1.0) + (0.9 * {ndcg_val:.4f}) = {final_score:.4f}")
                print("====================================================================")
                # ========== [디버깅 출력 종료] ==========
                
                reward_tensor[i, valid_response_length - 1] = final_score
            else:
                # Format 실패 시 0점 처리
                # ========== [디버깅 출력 시작] ==========
                print("===================== RAG REWARD (TRAIN) DEBUG =====================")
                print(f"[Sample {i}] Format FAIL (Final Score 0.0)")
                print(f"  - Full Text Start: {full_text[:50]}...")
                print("====================================================================")
                # ========== [디버깅 출력 종료] ==========
                reward_tensor[i, valid_response_length - 1] = 0.0

        return reward_tensor


# =============================================================================
# 3. Validation Reward Manager (for ray_trainer.py)
# =============================================================================

class RagEvalManagerPhase1:
    def __init__(self, tokenizer, num_examine):
        # rm_url 제거
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        """검증용: (Reward Tensor, nDCG List, Recall List) 3개 반환"""
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ndcg_list = []
        recall_list = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            full_text = self.tokenizer.decode(torch.cat([valid_prompt_ids, valid_response_ids]))

            # 1. Format Check
            format_score = simple_format_checker(full_text)
            
            ndcg_val = 0.0
            recall_val = 0.0
            final_score = 0.0

            if format_score > 0.0:
                # 2. Metric 계산
                try:
                    #extra_info = data_item.non_tensor_batch.get('extra_info', {})
                    reward_info = data_item.non_tensor_batch['reward_model']
                    retrieved_imgs = data_item.non_tensor_batch.get('retrievaled_images', [])
                    
                    retrievaled_base = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in retrieved_imgs]
                    
                    #ref_pages = extra_info.get('reference_page', [])
                    ref_pages = reward_info.get('reference_page', [])
                    if isinstance(ref_pages, np.ndarray): ref_pages = ref_pages.tolist()
                    #file_name = extra_info.get('file_name', 'unknown.pdf')
                    file_name = reward_info.get('file_name', 'unknown.pdf') 
                    reference_base = [f'{file_name.split(".pdf")[0]}_{p}' for p in ref_pages]

                    ndcg_val = ndcg(retrievaled_base, reference_base)
                    recall_val = recall_ret(retrievaled_base, reference_base)
                except Exception:
                    pass
                
                # 3. 점수 계산 (Train과 동일한 가중치)
                # Format: 0.1, nDCG: 0.9
                final_score = (0.1 * 1.0) + (0.9 * ndcg_val)

            reward_tensor[i, valid_response_length - 1] = final_score
            ndcg_list.append(ndcg_val)
            recall_list.append(recall_val)

        return reward_tensor, ndcg_list, recall_list