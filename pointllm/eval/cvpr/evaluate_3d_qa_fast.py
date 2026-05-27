#!/usr/bin/env python3
"""MO3D multi-perspective evaluation with an OpenAI LLM judge.

The evaluator reports:

1. Binary Accuracy for direct Yes/No conclusion matching.
2. Strict Accuracy for Yes/No conclusion plus reasoning consistency.
3. Semantic Accuracy for non-binary questions.

Results are grouped by task type and written to JSON.
"""

import argparse
import json
import re
import time
import asyncio
import aiohttp
from tqdm import tqdm
import os
from datetime import datetime
import random
import base64
from typing import Optional

API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY')
API_URL = "https://api.openai.com/v1/chat/completions"


# Shape Mating Classification Accuracy functions
def extract_sm_choice(text: str) -> Optional[str]:
    """Extract Shape Mating pair choice from text.
    Returns: '(1,2)', '(1,3)', '(2,3)', 'None', or None if not found.
    """
    if not text:
        return None
    match = re.search(r'\((\d),(\d)\)|None', text)
    return match.group(0) if match else None


def is_shape_mating_task(question: str) -> bool:
    """Detect if this is a Shape Mating task based on question content."""
    if not question:
        return False
    sm_keywords = [
        "mate", "interlock", "pair", "(1,2)", "(1,3)", "(2,3)",
        "Options:", "which pair"
    ]
    q_lower = question.lower()
    return any(kw.lower() in q_lower for kw in sm_keywords)

SYSTEM_PROMPT = """You are an impartial grader for a 3D-QA benchmark with multi-view image evidence.

CRITICAL: The images are the PRIMARY source of truth. Ground-truth text is a reference, but may not perfectly describe what's visible.

Evaluation criteria:
1. FIRST, examine the provided images carefully
2. Accept the model's answer if it reasonably describes what's visible in the images, even if it differs from ground-truth text
3. Accept answers that use synonyms, more specific terms, or additional visual details
4. Only reject if the answer clearly contradicts what's visible in ALL provided images

Examples of ACCEPTABLE answers (even if GT differs):
- Model says "plastic" when GT says "paper" → Accept if images show plastic-like appearance
- Model says "rectangular" when GT says "square" → Accept if visually reasonable
- Model provides more specific material/shape based on visual evidence → Accept

Return ONLY valid JSON: {"score":1, "reason":"≤20 words"} if visually grounded and reasonable, {"score":0, "reason":"≤20 words"} only if clearly contradicts visual evidence."""

HOLISTIC_SYSTEM_PROMPT = """You are an impartial grader for a 3D-QA benchmark with image evidence.

IMPORTANT: Images are the PRIMARY source of truth.

Evaluation:
1. Check if both answers have the same Yes/No conclusion
2. If they match → score=1
3. If they differ BUT model's answer is justified by visual evidence in images → score=1
4. Only score=0 if model's answer contradicts both ground-truth AND clear visual evidence

Return ONLY valid JSON: {"score":1, "reason":"≤15 words"} or {"score":0, "reason":"≤15 words"}."""

STRICT_SYSTEM_PROMPT = """You are a grader for a 3D-QA benchmark with multi-view image evidence.

CRITICAL: Images are the PRIMARY source of truth. Ground-truth is a reference but may not perfectly describe visual reality.

Evaluation criteria:
1. FIRST, examine the provided images carefully
2. Check if the Yes/No conclusion in "model_answer" is reasonable based on IMAGES
3. If model's conclusion matches ground-truth, verify reasoning is consistent with scene_description and images
4. If model's conclusion differs from ground-truth BUT is justified by visual evidence, ACCEPT IT
5. Accept reasoning with synonyms, more specific terms, or additional visual details

Examples of ACCEPTABLE answers:
- GT says "Yes, made of paper" but images show plastic appearance → Model saying "No, plastic" is acceptable
- GT says "square" but images show rectangular → Model saying "rectangular" is acceptable
- Model provides visually grounded reasoning even if GT differs → acceptable

Return ONLY valid JSON: {"score":1, "reason":"≤20 words"} if visually justified with consistent reasoning, {"score":0, "reason":"≤20 words"} only if contradicts clear visual evidence.
"""

def create_user_prompt(task_type, question, ground_truth, model_answer, scene_description):
    """境界マーキングを使った安全なプロンプト生成"""
    user_prompt = {
        "task_type": task_type,
        "question": question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "scene_description": scene_description
    }
    return json.dumps(user_prompt)

def create_scene_description(position_info):
    """position_infoから動的にscene_descriptionを生成"""
    scene_lines = []
    sorted_positions = sorted(position_info, key=lambda x: x.get('position_index', 0))
    
    for pos in sorted_positions:
        position_index = pos.get('position_index', 0)
        annotation = pos.get('annotation', 'Unknown object')
        scene_lines.append(f"  - Object at position {position_index}: {annotation}")
    
    return "\n".join(scene_lines)

def encode_image_to_data_url(image_path):
    """画像をbase64エンコードしてdata URLに変換"""
    try:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("ascii")
        
        # 拡張子から適切なMIMEタイプを判定
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        
        return f"data:{mime_type};base64,{image_data}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def extract_yes_no(text):
    """テキストからYes/Noを抽出（大文字小文字を考慮）- 拡張同義語対応"""
    if not text:
        return None
    
    text = text.strip().lower()
    
    # 適切な同義語セット（学術的で明確な表現のみ）
    YES_SYNONYMS = {
        'yes', 'Yes', 'absolutely', 'certainly', 'indeed'
    }
    
    NO_SYNONYMS = {
        'no', 'No'
    }
    
    # 最初の単語を抽出（句読点を除去）
    first_word = text.split()[0].strip('.,!?;:').lower() if text.split() else ""
    
    # 最初の単語での判定
    if first_word in YES_SYNONYMS:
        return 'yes'
    elif first_word in NO_SYNONYMS:
        return 'no'
    
    # より詳細な正規表現マッチング（全体テキストから）
    yes_pattern = r'\b(' + '|'.join(YES_SYNONYMS) + r')\b'
    no_pattern = r'\b(' + '|'.join(NO_SYNONYMS) + r')\b'
    
    if re.search(yes_pattern, text, re.IGNORECASE):
        return 'yes'
    elif re.search(no_pattern, text, re.IGNORECASE):
        return 'no'
    
    return None

def is_yes_no_question(question, ground_truth):
    """質問がYes/No質問かどうかを判定 - 正規表現拡張版"""
    question_lower = question.lower()
    ground_truth_lower = ground_truth.lower()
    
    # 拡張された質問パターン（正規表現）
    yes_no_question_pattern = r'^(is|are|do|does|did|can|could|will|would|should|have|has|had|may|might|must|ought)\b'
    
    # 質問パターンでの判定
    if re.match(yes_no_question_pattern, question_lower):
        return True
    
    # 追加の質問パターン
    additional_patterns = [
        r'\b(is there|are there|do any|does any|can any|will any)\b',
        r'\b(is it|are they|do they|does it|can it|will it)\b',
        r'\b(is this|are these|do these|does this|can this|will this)\b'
    ]
    
    for pattern in additional_patterns:
        if re.search(pattern, question_lower):
            return True
    
    # 正解がYes/Noで始まる場合
    if ground_truth_lower.startswith(('yes', 'no')):
        return True
        
    return False

def direct_holistic_qa_evaluation(ground_truth, model_answer):
    """holistic_qaタスクの直接Yes/No判定"""
    gt_yesno = extract_yes_no(ground_truth)
    ma_yesno = extract_yes_no(model_answer)
    
    if gt_yesno is None or ma_yesno is None:
        return None, f"Cannot extract Yes/No from: GT='{ground_truth}', MA='{model_answer}'"
    
    is_correct = gt_yesno == ma_yesno
    reason = f"Direct Yes/No comparison: GT='{gt_yesno}', MA='{ma_yesno}' → {'Match' if is_correct else 'Mismatch'}"
    
    return (1 if is_correct else 0), reason

async def make_api_request_with_retry(session, payload, headers, max_retries=3, sample_id="unknown"):
    """API リクエストをリトライ機能付きで実行"""
    
    for attempt in range(max_retries + 1):
        try:
            async with session.post(API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json(), None
                
                elif response.status == 500:
                    # 500エラーの場合はリトライ
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ API 500 error for sample {sample_id}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        print(f"❌ API 500 error for sample {sample_id} after {max_retries} retries: {error_text}")
                        return None, f"API 500 error after {max_retries} retries"
                
                elif response.status == 429:
                    # Rate limit エラーの場合
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) * 2 + random.uniform(0, 2)  # より長い待機時間
                        print(f"⚠️ Rate limit hit for sample {sample_id}, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return None, f"Rate limit exceeded after {max_retries} retries"
                
                else:
                    # その他のエラー
                    error_text = await response.text()
                    print(f"❌ API error {response.status} for sample {sample_id}: {error_text}")
                    return None, f"API error {response.status}: {error_text}"
                    
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Timeout for sample {sample_id}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                return None, f"Timeout after {max_retries} retries"
        
        except Exception as e:
            if attempt < max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Exception for sample {sample_id}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(wait_time)
                continue
            else:
                return None, f"Exception after {max_retries} retries: {e}"
    
    return None, "Unexpected error in API request retry logic"

async def evaluate_sample_async(session, sample_data, use_direct_evaluation=True, debug_mode=False, sample_index=None, max_images=4, quiet_mode=False):
    """単一サンプルの非同期評価 - 多角的評価フレームワーク（改良版エラーハンドリング + 画像対応）"""
    sample_id = sample_data.get('pilot_id', sample_data.get('id', 'unknown'))
    
    try:
        # 必要な情報を抽出
        question = sample_data.get('question', '')
        ground_truth = sample_data.get('ground_truth', '')
        # VLM結果のpredictionフィールド、または3DLLMのmodel_answerフィールドを使用
        model_answer = sample_data.get('prediction', sample_data.get('model_answer', ''))
        
        # Check if model_answer is empty
        if not model_answer or model_answer.strip() == '':
            return {
                'binary_accuracy': 0,
                'strict_accuracy': 0,
                'semantic_accuracy': 0
            }, "Empty model answer", "Model did not provide an answer"
        
        position_info = sample_data.get('position_info', [])
        num_objects = sample_data.get('num_point_clouds', len(position_info))
        # task_typeはmetadata内に含まれる場合がある
        metadata = sample_data.get('metadata', {})
        task_type = metadata.get('task_type', sample_data.get('task_type', 'unknown'))
        
        # 画像パスを取得
        image_paths = sample_data.get('image_paths', [])
        if image_paths and max_images > 0:
            image_paths = image_paths[:max_images]
        
        # scene_descriptionを動的に生成
        scene_description = create_scene_description(position_info)
        
        # 評価結果を格納する辞書
        judgements = {}
        
        # Yes/No質問の場合：Binary Accuracy + Strict Accuracy
        if is_yes_no_question(question, ground_truth) and use_direct_evaluation:
            # 1. Binary Accuracy の評価 (APIコールなし)
            binary_result, direct_reason = direct_holistic_qa_evaluation(ground_truth, model_answer)
            
            # None判定の安全処理
            if binary_result is None:
                if not quiet_mode:
                    print(f"Warning: Direct Binary evaluation failed for sample {sample_id}: {direct_reason}")
                    print(f"Falling back to GPT evaluation for both Binary and Strict...")
                # GPTバックアップ判定にフォールバック
                judgements['binary_accuracy'] = 0
                judgements['strict_accuracy'] = 0
                return judgements, f"Binary evaluation failed: {direct_reason}", direct_reason
            
            judgements['binary_accuracy'] = binary_result
            
            # デバッグモードでは直接判定の結果も表示
            if debug_mode and sample_index is not None and sample_index < 10:
                print(f"\n=== DEBUG Binary Accuracy evaluation (Sample {sample_id}) ===")
                print(f"Task Type: {task_type}")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Model Answer: {model_answer}")
                print(f"Binary result: {binary_result}")
                print(f"Direct reason: {direct_reason}")
            
            # 2. Strict Accuracy の評価
            if binary_result == 1:
                # 結論が合っている場合のみ、ReasoningをGPTで評価
                user_prompt = create_user_prompt(task_type, question, ground_truth, model_answer, scene_description)
                
                # メッセージコンテンツを構築（テキスト + 画像）
                message_content = [{"type": "text", "text": user_prompt}]
                
                # 画像を追加
                for img_path in image_paths:
                    data_url = encode_image_to_data_url(img_path)
                    if data_url:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"}
                        })
                
                payload = {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": [
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": message_content}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 100,
                    "temperature": 0
                }
                
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # 改良されたAPIリクエスト
                api_result, error_msg = await make_api_request_with_retry(session, payload, headers, sample_id=sample_id)
                
                if api_result is not None:
                    try:
                        response_text = api_result['choices'][0]['message']['content']
                        
                        # デバッグモード: Strict Accuracyの詳細出力
                        if debug_mode and sample_index is not None and sample_index < 10:
                            print(f"\n=== DEBUG Strict Accuracy evaluation (Sample {sample_id}) ===")
                            print(f"GPT Response:\n{response_text}")
                        
                        # JSON出力の解析
                        try:
                            json_response = json.loads(response_text)
                            strict_score = int(json_response.get('score', 0))
                            reason = json_response.get('reason', 'No reason provided')
                            
                            judgements['strict_accuracy'] = strict_score
                            
                            if debug_mode and sample_index is not None and sample_index < 10:
                                print(f"Parsed JSON: score={strict_score}, reason='{reason}'")
                                print("=" * 60)
                            
                            return judgements, f"Direct binary + GPT strict: {response_text}", reason
                            
                        except json.JSONDecodeError as e:
                            if not quiet_mode:
                                print(f"JSON decode error for strict evaluation (sample {sample_id}): {e}")
                            judgements['strict_accuracy'] = 0
                            return judgements, response_text, "JSON parse error in strict evaluation"
                            
                    except Exception as e:
                        if not quiet_mode:
                            print(f"Error in strict evaluation (sample {sample_id}): {e}")
                        judgements['strict_accuracy'] = 0
                        return judgements, f"Error: {e}", f"Error in strict evaluation: {e}"
                        
                else:
                    if not quiet_mode:
                        print(f"API request failed for strict evaluation (sample {sample_id}): {error_msg}")
                    judgements['strict_accuracy'] = 0
                    return judgements, f"API error: {error_msg}", "API error in strict evaluation"
                    
            else:
                # 結論が間違っていれば、Strict Accuracyは自動的に0
                judgements['strict_accuracy'] = 0
                response_text = "Direct binary (failed), strict skipped"
                reason = "Conclusion was incorrect, so strict check was skipped."
                
                if debug_mode and sample_index is not None and sample_index < 10:
                    print(f"\n=== DEBUG Strict Accuracy skipped (Sample {sample_id}) ===")
                    print(f"Reason: {reason}")
                    print("=" * 60)
                
                return judgements, response_text, reason
                
        elif is_yes_no_question(question, ground_truth) and not use_direct_evaluation:
            # Direct評価を無効にした場合：Yes/No質問をGPTで評価
            user_prompt = create_user_prompt(task_type, question, ground_truth, model_answer, scene_description)
            
            # メッセージコンテンツを構築（テキスト + 画像）
            message_content = [{"type": "text", "text": user_prompt}]
            
            # 画像を追加
            for img_path in image_paths:
                data_url = encode_image_to_data_url(img_path)
                if data_url:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"}
                    })
            
            # まずHOLISTIC_SYSTEM_PROMPTでBinary Accuracyを評価
            payload = {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": [
                    {"role": "system", "content": HOLISTIC_SYSTEM_PROMPT},
                    {"role": "user", "content": message_content}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 100,
                "temperature": 0
            }
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 改良されたAPIリクエスト
            api_result, error_msg = await make_api_request_with_retry(session, payload, headers, sample_id=sample_id)
            
            if api_result is not None:
                try:
                    response_text = api_result['choices'][0]['message']['content']
                    
                    json_response = json.loads(response_text)
                    binary_score = int(json_response.get('score', 0))
                    judgements['binary_accuracy'] = binary_score
                    
                    # Binary が正解の場合のみ、Strict Accuracyを評価
                    if binary_score == 1:
                        # STRICT_SYSTEM_PROMPTで評価（同じ画像付きメッセージを使用）
                        strict_payload = {
                            "model": "gpt-4o-mini-2024-07-18",
                            "messages": [
                                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                                {"role": "user", "content": message_content}
                            ],
                            "response_format": {"type": "json_object"},
                            "max_tokens": 100,
                            "temperature": 0
                        }
                        
                        strict_result, strict_error = await make_api_request_with_retry(session, strict_payload, headers, sample_id=sample_id)
                        
                        if strict_result is not None:
                            strict_response_text = strict_result['choices'][0]['message']['content']
                            strict_json = json.loads(strict_response_text)
                            strict_score = int(strict_json.get('score', 0))
                            judgements['strict_accuracy'] = strict_score
                            
                            return judgements, f"GPT Binary: {response_text}, GPT Strict: {strict_response_text}", f"Binary: {binary_score}, Strict: {strict_score}"
                        else:
                            judgements['strict_accuracy'] = 0
                            return judgements, response_text, f"Binary: {binary_score}, Strict: API error"
                    else:
                        judgements['strict_accuracy'] = 0
                        return judgements, response_text, f"Binary: {binary_score}, Strict: skipped"
                
                except json.JSONDecodeError as e:
                    judgements['binary_accuracy'] = 0
                    judgements['strict_accuracy'] = 0
                    return judgements, response_text, f"GPT Binary evaluation JSON parse error: {e}"
                    
            else:
                judgements['binary_accuracy'] = 0
                judgements['strict_accuracy'] = 0
                return judgements, f"API error: {error_msg}", "GPT Binary evaluation API error"
                
        else:
            # Yes/No以外の質問の評価 (Semantic Accuracy)
            user_prompt = create_user_prompt(task_type, question, ground_truth, model_answer, scene_description)
            
            # メッセージコンテンツを構築（テキスト + 画像）
            message_content = [{"type": "text", "text": user_prompt}]
            
            # 画像を追加
            for img_path in image_paths:
                data_url = encode_image_to_data_url(img_path)
                if data_url:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"}
                    })
            
            payload = {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message_content}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 100,
                "temperature": 0
            }
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 改良されたAPIリクエスト
            api_result, error_msg = await make_api_request_with_retry(session, payload, headers, sample_id=sample_id)
            
            if api_result is not None:
                try:
                    response_text = api_result['choices'][0]['message']['content']
                    
                    # デバッグモード: Semantic Accuracyの詳細出力
                    if debug_mode and sample_index is not None and sample_index < 10:
                        print(f"\n=== DEBUG Semantic Accuracy evaluation (Sample {sample_id}) ===")
                        print(f"Task Type: {task_type}")
                        print(f"Question: {question}")
                        print(f"Ground Truth: {ground_truth}")
                        print(f"Model Answer: {model_answer}")
                        print(f"GPT Response:\n{response_text}")
                    
                    # JSON出力の解析
                    try:
                        json_response = json.loads(response_text)
                        semantic_score = int(json_response.get('score', 0))
                        reason = json_response.get('reason', 'No reason provided')
                        
                        judgements['semantic_accuracy'] = semantic_score
                        
                        if debug_mode and sample_index is not None and sample_index < 10:
                            print(f"Parsed JSON: score={semantic_score}, reason='{reason}'")
                            print("=" * 60)
                        
                        return judgements, response_text, reason
                        
                    except json.JSONDecodeError as e:
                        if not quiet_mode:
                            print(f"JSON decode error for semantic evaluation (sample {sample_id}): {e}")
                        judgements['semantic_accuracy'] = 0
                        return judgements, response_text, "JSON parse error in semantic evaluation"
                        
                except Exception as e:
                    if not quiet_mode:
                        print(f"Error in semantic evaluation (sample {sample_id}): {e}")
                    judgements['semantic_accuracy'] = 0
                    return judgements, f"Error: {e}", f"Error in semantic evaluation: {e}"
                    
            else:
                if not quiet_mode:
                    print(f"API request failed for semantic evaluation (sample {sample_id}): {error_msg}")
                judgements['semantic_accuracy'] = 0
                return judgements, f"API error: {error_msg}", "API error in semantic evaluation"
                
    except Exception as e:
        if not quiet_mode:
            print(f"Error evaluating sample {sample_id}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        # エラーの場合は空の辞書を返す
        return {}, f"Error: {e}", f"Error: {e}"

async def evaluate_sample_async_with_retry(session, sample_data, use_direct_evaluation=True, debug_mode=False, sample_index=None, max_retries=3, max_images=4, quiet_mode=False):
    """リトライ機能付きの評価関数"""
    
    for attempt in range(max_retries + 1):
        try:
            result = await evaluate_sample_async(session, sample_data, use_direct_evaluation, debug_mode, sample_index, max_images, quiet_mode)
            return result
        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                base_wait = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait_time = base_wait + jitter
                
                sample_id = sample_data.get('pilot_id', sample_data.get('id', 'unknown'))
                print(f"⚠️ Retry {attempt + 1}/{max_retries} for sample {sample_id} after {wait_time:.1f}s (Error: {type(e).__name__})")
                
                await asyncio.sleep(wait_time)
            else:
                # 最後の試行でも失敗した場合
                sample_id = sample_data.get('pilot_id', sample_data.get('id', 'unknown'))
                print(f"❌ Failed after {max_retries} retries for sample {sample_id}: {e}")
                return {}, f"Failed after {max_retries} retries: {e}", f"Failed after retries: {e}"
    
    return {}, "Unexpected error in retry logic", "Unexpected error in retry logic"

def save_results_to_json(results, detailed_results, total_samples, overall_stats, start_time, end_time, input_file, output_file):
    """評価結果をJSONファイルに保存 - 多角的評価フレームワーク"""
    
    # タスクタイプごとの結果を計算
    task_type_results = {}
    for task_type, metrics in results.items():
        task_type_results[task_type] = {}
        for metric, judgements in metrics.items():
            if not judgements:  # データがない場合はスキップ
                continue
                
            correct_count = sum(judgements)
            total_count = len(judgements)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            task_type_results[task_type][metric] = {
                "accuracy": round(accuracy, 2),
                "correct": correct_count,
                "total": total_count
            }
    
    # 全体統計の計算
    overall_binary_accuracy = (overall_stats['binary_correct'] / overall_stats['binary_total'] * 100) if overall_stats['binary_total'] > 0 else 0
    overall_strict_accuracy = (overall_stats['strict_correct'] / overall_stats['strict_total'] * 100) if overall_stats['strict_total'] > 0 else 0
    overall_semantic_accuracy = (overall_stats['semantic_correct'] / overall_stats['semantic_total'] * 100) if overall_stats['semantic_total'] > 0 else 0
    overall_classification_accuracy = (overall_stats['classification_correct'] / overall_stats['classification_total'] * 100) if overall_stats.get('classification_total', 0) > 0 else None
    
    # 結果の構造化
    evaluation_results = {
        "evaluation_summary": {
            "input_file": input_file,
            "evaluation_framework": "Multi-Perspective (Binary, Strict, Semantic, Classification)",
            "overall_binary_accuracy": round(overall_binary_accuracy, 2),
            "overall_strict_accuracy": round(overall_strict_accuracy, 2),
            "overall_semantic_accuracy": round(overall_semantic_accuracy, 2),
            "overall_classification_accuracy": round(overall_classification_accuracy, 2) if overall_classification_accuracy is not None else None,
            "total_samples": total_samples,
            "binary_accuracy_samples": overall_stats['binary_total'],
            "strict_accuracy_samples": overall_stats['strict_total'],
            "semantic_accuracy_samples": overall_stats['semantic_total'],
            "classification_accuracy_samples": overall_stats.get('classification_total', 0),
            "model_used": "gpt-4o-mini-2024-07-18",
            "evaluation_start_time": start_time.isoformat(),
            "evaluation_end_time": end_time.isoformat(),
            "execution_time_seconds": int((end_time - start_time).total_seconds()),
            "execution_time_minutes": round((end_time - start_time).total_seconds() / 60, 2)
        },
        "task_type_results": task_type_results,
        "detailed_results": detailed_results
    }
    
    # JSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_file}")
    return evaluation_results

def save_metrics_only_to_json(results, overall_stats, output_file):
    """B, R, M, Cのメトリクスのみを保存（タスクタイプ別）"""
    
    # タスクタイプごとのメトリクスを計算
    metrics_by_task = {}
    
    # 全体のメトリクスを初期化
    overall_b_correct = 0
    overall_b_total = 0
    overall_r_correct = 0
    overall_r_total = 0
    overall_m_correct = 0
    overall_m_total = 0
    overall_c_correct = 0
    overall_c_total = 0
    
    for task_type, metrics in results.items():
        task_metrics = {}
        
        # Binary Accuracy (B)
        if 'binary_accuracy' in metrics and metrics['binary_accuracy']:
            b_correct = sum(metrics['binary_accuracy'])
            b_total = len(metrics['binary_accuracy'])
            task_metrics['B'] = round(b_correct / b_total, 4) if b_total > 0 else None
            overall_b_correct += b_correct
            overall_b_total += b_total
        else:
            task_metrics['B'] = None
        
        # Reasoning Accuracy (R)
        # Paper definition: reasoning is scored only among samples whose
        # binary/choice answer is correct. strict_accuracy stores zeroes for
        # binary-wrong samples, so use binary_correct as the denominator when
        # binary judgements are available.
        if 'strict_accuracy' in metrics and metrics['strict_accuracy']:
            r_correct = sum(metrics['strict_accuracy'])
            if 'binary_accuracy' in metrics and metrics['binary_accuracy']:
                r_total = sum(metrics['binary_accuracy'])
            else:
                r_total = len(metrics['strict_accuracy'])
            task_metrics['R'] = round(r_correct / r_total, 4) if r_total > 0 else None
            overall_r_correct += r_correct
            overall_r_total += r_total
        else:
            task_metrics['R'] = None
        
        # Semantic Accuracy (M)
        if 'semantic_accuracy' in metrics and metrics['semantic_accuracy']:
            m_correct = sum(metrics['semantic_accuracy'])
            m_total = len(metrics['semantic_accuracy'])
            task_metrics['M'] = round(m_correct / m_total, 4) if m_total > 0 else None
            overall_m_correct += m_correct
            overall_m_total += m_total
        else:
            task_metrics['M'] = None
        
        # Classification Accuracy (C) - Shape Mating用
        if 'classification_accuracy' in metrics and metrics['classification_accuracy']:
            c_correct = sum(metrics['classification_accuracy'])
            c_total = len(metrics['classification_accuracy'])
            task_metrics['C'] = round(c_correct / c_total, 4) if c_total > 0 else None
            overall_c_correct += c_correct
            overall_c_total += c_total
        else:
            task_metrics['C'] = None
        
        # サンプル数を追加
        total_samples = len(metrics.get('binary_accuracy', [])) + len(metrics.get('semantic_accuracy', []))
        if total_samples > 0:
            task_metrics['count'] = total_samples
            metrics_by_task[task_type] = task_metrics
    
    # 全体のメトリクスを計算
    overall_metrics = {
        'B': round(overall_b_correct / overall_b_total, 4) if overall_b_total > 0 else None,
        'R': round(overall_r_correct / overall_r_total, 4) if overall_r_total > 0 else None,
        'M': round(overall_m_correct / overall_m_total, 4) if overall_m_total > 0 else None,
        'C': round(overall_c_correct / overall_c_total, 4) if overall_c_total > 0 else None
    }
    
    # 最終的な結果構造
    metrics_results = {
        'overall': overall_metrics,
        **metrics_by_task
    }
    
    # JSONファイルに保存
    metrics_output_file = output_file.replace('_llm_eval.json', '_metrics.json')
    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metrics saved to {metrics_output_file}")
    return metrics_results

async def main():
    # 引数パーサーの設定
    parser = argparse.ArgumentParser(description="3D-QA Multi-Perspective Evaluation Script")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent requests")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (print first 10 evaluation responses)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode: only show tqdm progress bar")
    parser.add_argument("--holistic_direct", action="store_true", default=False, help="Use direct Yes/No comparison for Binary Accuracy evaluation")
    parser.add_argument("--no-holistic-direct", action="store_true", help="Disable direct Yes/No comparison (use GPT for all evaluations)")
    parser.add_argument("--max_images", type=int, default=4, help="Maximum number of images to attach per sample (default: 4, matching data generation)")
    
    args = parser.parse_args()
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # quietモードの設定
    quiet_mode = args.quiet
    
    # holistic_directの論理を調整（デフォルトでDirect評価を使用）
    use_direct_evaluation = args.holistic_direct or not args.no_holistic_direct
    if args.no_holistic_direct:
        use_direct_evaluation = False
    else:
        use_direct_evaluation = True  # デフォルトでDirect評価を使用
    
    # 入力ファイルパスの自動解決
    input_file = args.input_file
    
    # 現在のディレクトリでファイルが見つからない場合、親ディレクトリを探す
    if not os.path.exists(input_file):
        parent_path = os.path.join('..', input_file)
        if os.path.exists(parent_path):
            input_file = parent_path
            print(f"📁 Found input file at: {input_file}")
        else:
            # より上位のディレクトリも探す
            grandparent_path = os.path.join('..', '..', input_file)
            if os.path.exists(grandparent_path):
                input_file = grandparent_path
                print(f"📁 Found input file at: {input_file}")
            else:
                print(f"❌ Error: Input file not found: {args.input_file}")
                print(f"   Searched paths:")
                print(f"   - {args.input_file}")
                print(f"   - {parent_path}")
                print(f"   - {grandparent_path}")
                return
    
    # 出力ファイル名の設定
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 入力ファイルパスから出力先を判定
        base_output_dir = os.path.join("outputs", "llm_eval")
        if "vlm_results" in input_file or "vlm" in input_file.lower():
            # VLM結果の場合
            output_dir = os.path.join(base_output_dir, "vlm")
        elif "outputs" in input_file or "evaluation" in input_file:
            # 3DLLM (PointLLM等)結果の場合
            output_dir = os.path.join(base_output_dir, "3dllm")
        else:
            # その他の場合はベースディレクトリに保存
            output_dir = base_output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{base_name}_llm_eval.json")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # 結果を保存する辞書（階層構造: task_type -> metric -> scores）
    results = {}
    detailed_results = []
    total_samples = 0
    error_count = 0
    debug_responses = []
    direct_evaluation_count = 0
    json_parse_error_count = 0
    gpt_evaluation_count = 0
    strict_evaluation_count = 0
    
    start_time = datetime.now()
    if not quiet_mode:
        print(f"Starting Multi-Perspective evaluation of {input_file} using GPT-4o-mini-2024-07-18...")
        print("📊 Evaluation Framework:")
        print("  - Binary Accuracy: Yes/No conclusion matching")
        print("  - Strict Accuracy: Yes/No conclusion + reasoning consistency")
        print("  - Semantic Accuracy: Meaning-based evaluation for complex questions")
    print("🔧 Error Handling:")
    print("  - Exponential backoff retry for API failures")
    print("  - Automatic retry for 500 and 429 errors")
    print("  - Improved rate limiting")
    
    if args.debug:
        print("🐛 Debug mode: Will print first 10 evaluation responses")
    if use_direct_evaluation:
        print("🚀 Using direct Yes/No comparison for Binary Accuracy evaluation")
    else:
        print("🤖 Using GPT evaluation for all assessments (including Binary Accuracy)")
    
    print(f"⚙️ Configuration: batch_size={args.batch_size}, enhanced_error_handling=True")
    print(f"📸 Maximum images per sample: {args.max_images} (matching data generation: 4 views per object)")
    
    # 入力ファイルの処理
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
        # データ形式を判定して正規化
        if isinstance(json_data, list):
            # VLM形式（直接配列）
            data = json_data
        elif isinstance(json_data, dict) and 'results' in json_data:
            # PointLLM形式（results配列を持つ）
            data = json_data['results']
            # image_refsをimage_pathsに変換
            for sample in data:
                if 'image_paths' not in sample:
                    # metadataの中のimage_refsをチェック
                    metadata = sample.get('metadata', {})
                    image_refs = sample.get('image_refs', metadata.get('image_refs', {}))
                    
                    if image_refs:
                        # image_refsから画像パスをフラット化（各オブジェクトから1枚ずつ順番に）
                        image_paths = []
                        obj_ids = list(image_refs.keys())
                        max_views_per_obj = 4  # 各オブジェクトから最大4枚
                        
                        # 各ビュー角度ごとに、全オブジェクトから1枚ずつ取得
                        for view_idx in range(max_views_per_obj):
                            for obj_id in obj_ids:
                                paths = image_refs[obj_id]
                                if view_idx < len(paths):
                                    image_paths.append(paths[view_idx])
                        
                        sample['image_paths'] = image_paths
        else:
            print("Error: Unsupported JSON format. Expected a list or a dict with 'results' key.")
            return
        
        # サンプル数制限
        if args.max_samples:
            data = data[:args.max_samples]
        
        # 非同期でバッチ処理
        timeout = aiohttp.ClientTimeout(total=60)  # 60秒のタイムアウト（倍増）
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # tqdmプログレスバーの設定（quietモード時のみ表示）
            total_batches = (len(data) + args.batch_size - 1) // args.batch_size
            pbar = tqdm(total=len(data), desc="Evaluating", unit="sample", disable=(not quiet_mode))
            
            for i in range(0, len(data), args.batch_size):
                batch = data[i:i + args.batch_size]
                
                # バッチ内の各サンプルを並行処理（リトライ機能付き）
                tasks = [evaluate_sample_async_with_retry(session, sample, use_direct_evaluation=use_direct_evaluation, debug_mode=args.debug, sample_index=i + idx, max_images=args.max_images, quiet_mode=quiet_mode) 
                        for idx, sample in enumerate(batch)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 結果を集計
                for sample, result in zip(batch, batch_results):
                    # 例外が発生した場合の処理
                    if isinstance(result, Exception):
                        sample_id = sample.get('pilot_id', sample.get('id', f'sample_{total_samples}'))
                        print(f"Exception occurred for sample {sample_id}: {result}")
                        judgements, response_text, reason = {}, f"Exception: {result}", f"Exception: {result}"
                        error_count += 1
                    else:
                        judgements, response_text, reason = result
                        if "Error:" in response_text or "API error:" in response_text or "Failed after" in response_text:
                            error_count += 1
                        
                        # 評価方法の統計カウント
                        if "Direct binary" in response_text or "Direct Yes/No evaluation" in response_text:
                            direct_evaluation_count += 1
                        if "GPT strict" in response_text or "Direct binary + GPT strict" in response_text:
                            strict_evaluation_count += 1
                        if "semantic_accuracy" in judgements:
                            gpt_evaluation_count += 1
                        if "JSON parse error" in response_text:
                            json_parse_error_count += 1
                    
                    # task_typeをmetadataから取得（VLM結果の形式に対応）
                    metadata = sample.get('metadata', {})
                    task_type = metadata.get('task_type', sample.get('task_type', 'unknown'))
                    sample_id = sample.get('pilot_id', sample.get('id', f'sample_{total_samples}'))
                    
                    # task_typeの辞書を初期化
                    if task_type not in results:
                        results[task_type] = {
                            'binary_accuracy': [],
                            'strict_accuracy': [],
                            'semantic_accuracy': [],
                            'classification_accuracy': []  # For Shape Mating
                        }
                    
                    # 評価結果を対応するリストに追加
                    for metric, score in judgements.items():
                        if score is not None:
                            results[task_type][metric].append(score)
                    
                    # Shape Mating Classification Accuracy の計算
                    question = sample.get('question', '')
                    ground_truth = sample.get('ground_truth', '')
                    model_answer = sample.get('prediction', sample.get('model_answer', ''))
                    is_sm = task_type == "shape_mating" or is_shape_mating_task(question)
                    sm_correct = None
                    
                    if task_type == "shape_mating" and is_sm:
                        gt_choice = extract_sm_choice(ground_truth)
                        pred_choice = extract_sm_choice(model_answer)
                        sm_correct = 1 if gt_choice is not None and gt_choice == pred_choice else 0
                        results[task_type]['classification_accuracy'].append(sm_correct)
                    
                    # 詳細結果を保存
                    detailed_results.append({
                        "sample_id": sample_id,
                        "task_type": task_type,
                        "judgements": judgements,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_answer": model_answer,
                        "is_shape_mating": is_sm,
                        "sm_classification_correct": sm_correct,
                        "evaluation_response": response_text,
                        "reason": reason
                    })
                    
                    total_samples += 1
                
                # 進捗更新
                pbar.update(len(batch))
                
                # 進捗表示（quietモードでは非表示）
                if not quiet_mode:
                    processed = min(i + args.batch_size, len(data))
                    print(f"Processed {processed}/{len(data)} samples (Errors: {error_count})")
                
                # API制限対策（改善された待機）
                # エラー発生率に応じて待機時間を調整
                error_rate = error_count / total_samples if total_samples > 0 else 0
                if error_rate > 0.1:  # 10%以上のエラー率の場合
                    await asyncio.sleep(0.3)  # 長めの待機
                elif error_rate > 0.05:  # 5%以上のエラー率の場合
                    await asyncio.sleep(0.2)  # 中程度の待機
                else:
                    await asyncio.sleep(0.1)  # 標準的な待機
            
            # プログレスバーを閉じる
            pbar.close()
    
    end_time = datetime.now()
    
    # 精度の計算と結果の表示
    overall_binary_correct = 0
    overall_binary_total = 0
    overall_strict_correct = 0
    overall_strict_total = 0
    overall_semantic_correct = 0
    overall_semantic_total = 0
    overall_classification_correct = 0
    overall_classification_total = 0
    
    # task_typeごとの詳細な正解率を計算
    for task_type, metrics in results.items():
        if not quiet_mode:
            print(f"\n{task_type}:")
            print("-" * 40)
        
        for metric, judgements in metrics.items():
            if not judgements:  # データがない場合はスキップ
                continue
                
            correct_count = sum(judgements)
            total_count = len(judgements)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            if not quiet_mode:
                print(f"  {metric}: {accuracy:.2f}% ({correct_count}/{total_count})")
            
            # 全体統計に加算
            if metric == 'binary_accuracy':
                overall_binary_correct += correct_count
                overall_binary_total += total_count
            elif metric == 'strict_accuracy':
                overall_strict_correct += correct_count
                overall_strict_total += total_count
            elif metric == 'semantic_accuracy':
                overall_semantic_correct += correct_count
                overall_semantic_total += total_count
            elif metric == 'classification_accuracy':
                overall_classification_correct += correct_count
                overall_classification_total += total_count
    
    if not quiet_mode:
        print("\n" + "="*70)
        print("--- Multi-Perspective Evaluation Results ---")
        print("\n" + "="*70)
        print("--- Overall Performance Summary ---")
        
        # 全体の正解率を計算
        if overall_binary_total > 0:
            overall_binary_accuracy = (overall_binary_correct / overall_binary_total) * 100
            print(f"Overall Binary Accuracy:   {overall_binary_accuracy:.2f}% ({overall_binary_correct}/{overall_binary_total})")
        
        if overall_strict_total > 0:
            overall_strict_accuracy = (overall_strict_correct / overall_strict_total) * 100
            print(f"Overall Strict Accuracy:   {overall_strict_accuracy:.2f}% ({overall_strict_correct}/{overall_strict_total})")
        
        if overall_semantic_total > 0:
            overall_semantic_accuracy = (overall_semantic_correct / overall_semantic_total) * 100
            print(f"Overall Semantic Accuracy: {overall_semantic_accuracy:.2f}% ({overall_semantic_correct}/{overall_semantic_total})")
        
        if overall_classification_total > 0:
            overall_classification_accuracy = (overall_classification_correct / overall_classification_total) * 100
            print(f"Overall Classification Accuracy (SM): {overall_classification_accuracy:.2f}% ({overall_classification_correct}/{overall_classification_total})")
    
    print("="*70)
    print(f"📊 Evaluation completed. Total samples processed: {total_samples}")
    print(f"❌ Errors encountered: {error_count}")
    print(f"🚀 Direct evaluations (Binary Accuracy): {direct_evaluation_count}")
    print(f"🎯 Strict evaluations (Yes/No + Reasoning): {strict_evaluation_count}")
    print(f"🤖 Semantic evaluations (GPT-based): {gpt_evaluation_count}")
    print(f"📝 JSON parse errors: {json_parse_error_count}")
    
    # 改善効果の計算
    if total_samples > 0:
        direct_success_rate = (direct_evaluation_count / total_samples) * 100
        strict_success_rate = (strict_evaluation_count / total_samples) * 100
        semantic_success_rate = (gpt_evaluation_count / total_samples) * 100
        parse_error_rate = (json_parse_error_count / total_samples) * 100
        
        print(f"📈 Direct evaluation coverage: {direct_success_rate:.1f}%")
        print(f"🎯 Strict evaluation coverage: {strict_success_rate:.1f}%") 
        print(f"🤖 Semantic evaluation coverage: {semantic_success_rate:.1f}%")
        print(f"⚠️ JSON parse error rate: {parse_error_rate:.1f}%")
    
    print(f"⏱️ Execution time: {(end_time - start_time).total_seconds():.1f} seconds")
    
    # 結果をJSONファイルに保存
    overall_stats = {
        'binary_correct': overall_binary_correct,
        'binary_total': overall_binary_total,
        'strict_correct': overall_strict_correct,
        'strict_total': overall_strict_total,
        'semantic_correct': overall_semantic_correct,
        'semantic_total': overall_semantic_total,
        'classification_correct': overall_classification_correct,
        'classification_total': overall_classification_total
    }
    save_results_to_json(results, detailed_results, total_samples, overall_stats, 
                        start_time, end_time, input_file, args.output)
    
    # B, R, Mメトリクスのみを保存
    save_metrics_only_to_json(results, overall_stats, args.output)

if __name__ == "__main__":
    asyncio.run(main()) 
