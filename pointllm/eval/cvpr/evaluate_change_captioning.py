#!/usr/bin/env python3
"""
Change Captioning & Shape Mating Evaluation Script
Evaluates tasks without using images (LLM-based evaluation only)

Change Captioning Tasks:
- verify: Yes/No verification with reasoning
- delta_caption: Geometric change description

Shape Mating Tasks:
- shape_mating: Select which pairs can mate (single-task mode)
"""

import os
import json
import argparse
import asyncio
import aiohttp
import re
from datetime import datetime
from tqdm import tqdm

# OpenAI API設定
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"

# ============================================================================
# Evaluation Prompts
# ============================================================================

VERIFY_SYSTEM_PROMPT = """You are an impartial grader for a 3D shape verification task.

Evaluate whether the model's answer correctly verifies if requirements are met.

Evaluation criteria:
1. Binary (B): Does the Yes/No conclusion match the ground truth?
2. Reasoning (R): Is the reasoning factually consistent with the requirements and ground truth?

Return ONLY valid JSON: {"B": 0|1, "R": 0|1, "reason": "brief explanation"}
Keep reason under 20 words."""

DELTA_CAPTION_SYSTEM_PROMPT = """You are an impartial grader for a geometric change description task.

Evaluate the model's description of geometric changes using a 10-point scale.

INSTRUCTIONS:
1. Break down the ground truth into individual geometric modification items
2. Check how many of these items are correctly captured by the model's answer
3. Accept semantically equivalent descriptions even if phrasing differs
4. If the model's answer CONTRADICTS any ground truth item, return M=0 immediately
5. Otherwise, score based on coverage:
   - M=10: All items correctly described (exact or semantically equivalent)
   - M=7-9: Most items correct, minor omissions
   - M=4-6: About half the items correct
   - M=1-3: Few items correct, mostly incomplete
   - M=0: Major contradictions OR almost no correct items

Return ONLY valid JSON: {"M": 0-10, "reason": "brief explanation"}
Keep reason under 30 words."""

SHAPE_MATING_SYSTEM_PROMPT = """You are an impartial grader for a shape mating task.

Evaluate whether the model correctly identified which pairs can mate (fit together geometrically).

Evaluation criteria:
1. Selection (S): Did the model select the correct pair(s) or "None"?
2. Reasoning (R): Is the reasoning about why pairs can/cannot mate factually consistent?

Return ONLY valid JSON: {"S": 0|1, "R": 0|1, "reason": "brief explanation"}
Keep reason under 20 words."""

# ============================================================================
# Helper Functions
# ============================================================================

def extract_mating_pairs(text):
    """Extract mating pairs from text: (1,2), (1,3), (2,3), or None
    Also handles Answer: A/B/C/D format where:
    - A = (1,2), B = (1,3), C = (2,3), D = None
    """
    text_lower = text.lower()
    pairs = []
    
    # First, try to extract A/B/C/D format (Answer: X)
    answer_match = re.search(r'answer:\s*([a-d])', text_lower)
    if answer_match:
        choice = answer_match.group(1).upper()
        choice_to_pair = {'A': '(1,2)', 'B': '(1,3)', 'C': '(2,3)', 'D': 'None'}
        return [choice_to_pair.get(choice, 'None')]
    
    # Check for standalone A/B/C/D at the beginning
    standalone_match = re.match(r'^([a-d])\b', text.strip(), re.IGNORECASE)
    if standalone_match:
        choice = standalone_match.group(1).upper()
        choice_to_pair = {'A': '(1,2)', 'B': '(1,3)', 'C': '(2,3)', 'D': 'None'}
        return [choice_to_pair.get(choice, 'None')]
    
    # Check for "None"
    if re.search(r'\bnone\b', text_lower) and not re.search(r'\([\d,]+\)', text):
        return ['None']
    
    # Extract pairs like (1,2), (1,3), (2,3)
    pair_pattern = r'\((\d+)\s*,\s*(\d+)\)'
    matches = re.findall(pair_pattern, text)
    
    for match in matches:
        pair_str = f"({match[0]},{match[1]})"
        if pair_str not in pairs:
            pairs.append(pair_str)
    
    # If no pairs found but not explicitly "None", return empty list
    return pairs if pairs else []

def extract_selection(text):
    """Extract A or B selection from text"""
    text_lower = text.lower()
    
    # Pattern matching for clear statements
    patterns = [
        r'\bcandidates?\s+([ab])\b',
        r'\boption\s+([ab])\b',
        r'\b([ab])\s*\.',
        r'^([ab])\b',
        r'\bfirst\s+(?:shape|option|candidate)\b',  # A
        r'\bsecond\s+(?:shape|option|candidate)\b',  # B
        r'\bthird\s+(?:shape|option|candidate)\b',   # B (in context of question)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            selection = match.group(1).upper()
            if selection in ['A', 'B']:
                return selection
    
    # Contextual extraction
    if 'second shape' in text_lower or 'candidate a' in text_lower:
        return 'A'
    if 'third shape' in text_lower or 'candidate b' in text_lower:
        return 'B'
    
    return None

def extract_yes_no(text):
    """Extract Yes or No from text"""
    text_lower = text.lower().strip()
    
    # Direct match at the beginning
    if text_lower.startswith('yes'):
        return 'Yes'
    if text_lower.startswith('no'):
        return 'No'
    
    # Pattern matching
    yes_patterns = [r'\byes\b', r'\bcorrect\b', r'\btrue\b']
    no_patterns = [r'\bno\b', r'\bincorrect\b', r'\bfalse\b']
    
    has_yes = any(re.search(p, text_lower) for p in yes_patterns)
    has_no = any(re.search(p, text_lower) for p in no_patterns)
    
    if has_yes and not has_no:
        return 'Yes'
    if has_no and not has_yes:
        return 'No'
    
    return None

# ============================================================================
# API Request Functions
# ============================================================================

async def make_api_request_with_retry(session, payload, headers, max_retries=3):
    """Make API request with exponential backoff retry"""
    import random
    
    for attempt in range(max_retries + 1):
        try:
            async with session.post(API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result, None
                elif response.status in [429, 500, 502, 503, 504]:
                    # Retryable errors
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        return None, f"HTTP {response.status}: {error_text}"
                else:
                    error_text = await response.text()
                    return None, f"HTTP {response.status}: {error_text}"
        except Exception as e:
            if attempt < max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            else:
                return None, f"Exception after {max_retries} retries: {e}"
    
    return None, "Unexpected error in API request retry logic"

# ============================================================================
# Task-specific Evaluation Functions
# ============================================================================

async def evaluate_verify_task(session, sample_data, debug_mode=False, sample_index=None, quiet_mode=False):
    """Evaluate verify task (Yes/No with reasoning)"""
    question = sample_data.get('question', '')
    ground_truth = sample_data.get('ground_truth', '')
    model_answer = sample_data.get('prediction', sample_data.get('model_answer', ''))
    model_reasoning = sample_data.get('reasoning', '')
    metadata = sample_data.get('metadata', {}) or {}
    gt_reasoning = metadata.get('reason_gt') or metadata.get('rationale') or metadata.get('why') or ''
    ground_truth_for_judge = ground_truth
    if gt_reasoning:
        ground_truth_for_judge = f"{ground_truth}\nReason: {gt_reasoning}"
    model_answer_for_judge = model_answer
    if model_reasoning:
        model_answer_for_judge = f"{model_answer}\nReasoning: {model_reasoning}"
    
    # Check if model_answer is empty
    if not model_answer or model_answer.strip() == '':
        if debug_mode and sample_index is not None and sample_index < 10:
            if not quiet_mode:
                print(f"\n=== DEBUG verify (Sample {sample_index}) ===")
                print("Model answer is empty. Returning B=0, R=0")
        return {'B': 0, 'R': 0}, "Empty model answer", "Model did not provide an answer"
    
    # Extract Yes/No from both
    gt_decision = extract_yes_no(ground_truth)
    model_decision = extract_yes_no(model_answer)
    
    # Binary evaluation (direct comparison)
    binary_score = 1 if (gt_decision and model_decision and gt_decision == model_decision) else 0
    
    # Reasoning evaluation (using GPT)
    user_prompt = f"""Question: {question}

Ground Truth: {ground_truth_for_judge}

Model Answer: {model_answer_for_judge}

Evaluate the model's reasoning quality."""
    
    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    result, error = await make_api_request_with_retry(session, payload, headers)
    
    if result:
        try:
            response_text = result['choices'][0]['message']['content']
            response_json = json.loads(response_text)
            
            # Use GPT's judgement, fallback to regex extraction if GPT didn't provide
            binary_score_gpt = response_json.get('B', binary_score)
            reasoning_score = response_json.get('R', binary_score)
            reason = response_json.get('reason', 'No reason provided')
            
            if debug_mode and sample_index is not None and sample_index < 10:
                print(f"\n=== DEBUG verify (Sample {sample_index}) ===")
                print(f"GT: {ground_truth_for_judge[:100]}")
                print(f"Model: {model_answer_for_judge[:100]}")
                print(f"Binary (regex): {binary_score}, Binary (GPT): {binary_score_gpt}")
                print(f"Reasoning: {reasoning_score}")
                print(f"Reason: {reason}")
            
            return {
                'B': binary_score_gpt,  # Use GPT's judgement
                'R': reasoning_score
            }, response_text, reason
            
        except json.JSONDecodeError as e:
            if not quiet_mode:
                print(f"JSON decode error: {e}")
            return {'B': binary_score, 'R': 0}, response_text, "JSON parse error"
    else:
        if not quiet_mode:
            print(f"API error: {error}")
        return {'B': binary_score, 'R': 0}, f"API error: {error}", "API error"

async def evaluate_delta_caption_task(session, sample_data, debug_mode=False, sample_index=None, quiet_mode=False):
    """Evaluate delta_caption task (geometric change description)"""
    original_question = sample_data.get('question', '')
    ground_truth = sample_data.get('ground_truth', '')
    model_answer = sample_data.get('prediction', sample_data.get('model_answer', ''))
    
    # Check if model_answer is empty
    if not model_answer or model_answer.strip() == '':
        if debug_mode and sample_index is not None and sample_index < 10:
            if not quiet_mode:
                print(f"\n=== DEBUG delta_caption (Sample {sample_index}) ===")
                print("Model answer is empty. Returning M=0")
        return {'M': 0}, "Empty model answer", "Model did not provide an answer"
    
    question = original_question.strip()
    
    user_prompt = f"""Question: {question}

Ground Truth: {ground_truth}

Model Answer: {model_answer}

Evaluate if the model's geometric change description is accurate and complete."""
    
    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {"role": "system", "content": DELTA_CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    result, error = await make_api_request_with_retry(session, payload, headers)
    
    if result:
        try:
            response_text = result['choices'][0]['message']['content']
            response_json = json.loads(response_text)
            
            semantic_score = response_json.get('M', 0)
            reason = response_json.get('reason', 'No reason provided')
            
            # Validate score is in 0-10 range
            if not isinstance(semantic_score, (int, float)):
                semantic_score = 0
            semantic_score = max(0, min(10, semantic_score))  # Clamp to [0, 10]
            
            if debug_mode and sample_index is not None and sample_index < 10:
                if not quiet_mode:
                    print(f"\n=== DEBUG delta_caption (Sample {sample_index}) ===")
                    print(f"GT: {ground_truth[:100]}")
                    print(f"Model: {model_answer[:100]}")
                    print(f"Semantic (0-10): {semantic_score}")
                    print(f"Reason: {reason}")
            
            return {
                'M': semantic_score
            }, response_text, reason
            
        except json.JSONDecodeError as e:
            if not quiet_mode:
                print(f"JSON decode error: {e}")
            return {'M': 0}, response_text, "JSON parse error"
    else:
        if not quiet_mode:
            print(f"API error: {error}")
        return {'M': 0}, f"API error: {error}", "API error"

async def evaluate_shape_mating_task(session, sample_data, debug_mode=False, sample_index=None, quiet_mode=False):
    """Evaluate shape_mating task (pair selection with reasoning)"""
    question = sample_data.get('question', '')
    ground_truth = sample_data.get('ground_truth', '')
    model_answer = sample_data.get('prediction', sample_data.get('model_answer', ''))
    model_reasoning = sample_data.get('reasoning', '')  # Get reasoning from multi-turn inference
    gt_answer = sample_data.get('answer', [])
    
    # Check if model_answer is empty
    if not model_answer or model_answer.strip() == '':
        if debug_mode and sample_index is not None and sample_index < 10:
            if not quiet_mode:
                print(f"\n=== DEBUG shape_mating (Sample {sample_index}) ===")
                print("Model answer is empty. Returning S=0, R=0")
        return {'S': 0, 'R': 0}, "Empty model answer", "Model did not provide an answer"
    
    # If answer field is missing or empty, extract from ground_truth
    if not gt_answer:
        gt_answer = extract_mating_pairs(ground_truth)
    
    # Extract pairs from both
    if isinstance(gt_answer, list):
        gt_pairs = set(gt_answer)
    else:
        gt_pairs = set([gt_answer])
    
    model_pairs = set(extract_mating_pairs(model_answer))
    
    # Selection evaluation (set comparison)
    selection_score = 1 if gt_pairs == model_pairs else 0
    
    # Reasoning evaluation logic:
    # - If selection is WRONG (S=0): R=0 automatically (can't have correct reasoning for wrong answer)
    # - If selection is CORRECT (S=1): Use GPT to evaluate reasoning quality
    # - If no reasoning provided: R=0
    
    if selection_score == 0:
        # Wrong selection -> reasoning is not evaluated (R=None, not R=0)
        if debug_mode and sample_index is not None and sample_index < 10:
            if not quiet_mode:
                print(f"\n=== DEBUG shape_mating (Sample {sample_index}) ===")
                print(f"GT Pairs: {gt_pairs}, Model Pairs: {model_pairs}")
                print(f"Selection WRONG (S=0), R not evaluated (N/A)")
                print(f"Model reasoning: {model_reasoning[:100] if model_reasoning else 'None'}...")
        return {'S': 0, 'R': None}, "Selection wrong", "Selection is incorrect, reasoning not evaluated"
    
    # Selection is correct (S=1), now evaluate reasoning
    if not model_reasoning or model_reasoning.strip() == '':
        if debug_mode and sample_index is not None and sample_index < 10:
            if not quiet_mode:
                print(f"\n=== DEBUG shape_mating (Sample {sample_index}) ===")
                print(f"GT Pairs: {gt_pairs}, Model Pairs: {model_pairs}")
                print(f"Selection CORRECT (S=1), but no reasoning provided (R=0)")
        return {'S': 1, 'R': 0}, "No reasoning", "Selection correct but no reasoning provided"
    
    # Get ground truth reasoning (why_ref) if available
    why_ref = sample_data.get('why_ref', {})
    gt_reasoning = ""
    if why_ref:
        # Get the correct reasoning based on answer
        if gt_answer == ['None']:
            # For None answer, all pairs don't fit - use any explanation
            gt_reasoning = "; ".join([f"{k}: {v}" for k, v in why_ref.items()])
        else:
            # Get reasoning for the correct pair
            correct_pair = gt_answer[0] if gt_answer else ''
            gt_reasoning = why_ref.get(correct_pair, '')
    
    # Build prompt
    if gt_reasoning:
        # Compare against ground truth reasoning
        user_prompt = f"""Question: {question}

Correct Answer: {list(gt_answer)}
Ground Truth Reasoning: {gt_reasoning}

Model Answer: {model_answer}
Model Reasoning: {model_reasoning}

The model selected the CORRECT answer. Evaluate if its reasoning is SEMANTICALLY SIMILAR to the ground truth reasoning.
- Does the model's reasoning convey the same meaning as the ground truth?
- Is it explaining WHY the pair fits (or doesn't fit) correctly?
- Generic/template responses like "matching profiles" without specific details should get R=0.

Return ONLY: {{"R": 0 or 1, "reason": "brief explanation"}}
R=1 if reasoning is semantically similar to ground truth, R=0 if it's wrong, vague, or generic."""
    else:
        # No ground truth reasoning available, use basic evaluation
        user_prompt = f"""Question: {question}

Correct Answer: {list(gt_answer)}
Model Answer: {model_answer}
Model Reasoning: {model_reasoning}

The model selected the CORRECT answer. Evaluate if its reasoning is correct and specific.
- Does it explain WHY the selected pair fits (or why no pairs fit)?
- Generic/template responses without specific analysis should get R=0.

Return ONLY: {{"R": 0 or 1, "reason": "brief explanation"}}
R=1 if reasoning is correct and specific, R=0 if wrong or generic."""
    
    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {"role": "system", "content": "You evaluate shape mating reasoning. Return only valid JSON."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    result, error = await make_api_request_with_retry(session, payload, headers)
    
    if result:
        try:
            response_text = result['choices'][0]['message']['content']
            response_json = json.loads(response_text)
            
            reasoning_score = response_json.get('R', 0)
            reason = response_json.get('reason', 'No reason provided')
            
            if debug_mode and sample_index is not None and sample_index < 10:
                if not quiet_mode:
                    print(f"\n=== DEBUG shape_mating (Sample {sample_index}) ===")
                    print(f"GT Pairs: {gt_pairs}, Model Pairs: {model_pairs}")
                    print(f"Selection CORRECT (S=1)")
                    print(f"GT reasoning: {gt_reasoning[:100] if gt_reasoning else 'N/A'}...")
                    print(f"Model reasoning: {model_reasoning[:100]}...")
                    print(f"GPT R score: {reasoning_score}")
                    print(f"GPT reason: {reason}")
            
            return {
                'S': 1,  # Selection is correct
                'R': reasoning_score
            }, response_text, reason
            
        except json.JSONDecodeError as e:
            if not quiet_mode:
                print(f"JSON decode error: {e}")
            return {'S': 1, 'R': 0}, response_text, "JSON parse error"
    else:
        if not quiet_mode:
            print(f"API error: {error}")
        return {'S': 1, 'R': 0}, f"API error: {error}", "API error"

# ============================================================================
# Main Evaluation Function
# ============================================================================

async def evaluate_sample_async(session, sample_data, debug_mode=False, sample_index=None, quiet_mode=False):
    """Evaluate a single sample based on its task type"""
    task = sample_data.get('task', sample_data.get('metadata', {}).get('task', 'unknown'))
    
    # Auto-detect task type from question if task_type is missing, unknown, or not set
    question = sample_data.get('question', '')
    question_lower = question.lower()
    
    if not task or task == 'unknown':
        # Shape mating: "Which pairs can mate?" or "Which pair can mate?" (select one mode)
        # Also detect "Find the one pair that fits together"
        if ('which pair' in question_lower and 'can mate' in question_lower) or \
           ('find the one pair' in question_lower and 'fits' in question_lower) or \
           ('options: a, b, c, d' in question_lower and '(1,2)' in question_lower):
            task = 'shape_mating'
        # Delta caption: shape modification prompts
        elif 'shape modification model' in question_lower or 'what text prompt would you provide' in question_lower:
            task = 'delta_caption'
        # Verify: "Do all of these requirements hold" or "Does the ... object satisfy"
        elif 'do all of these requirements hold' in question_lower or 'does the first object satisfy' in question_lower or 'does the second object satisfy' in question_lower:
            task = 'verify'
    
    # Debug: Show detected task type
    if debug_mode and sample_index is not None and sample_index < 10:
        if not quiet_mode:
            print(f"\n=== Sample {sample_index}: Detected task = {task} ===")
    
    # Evaluate based on task type and return result with detected task
    result = None
    if task == 'verify':
        result = await evaluate_verify_task(session, sample_data, debug_mode, sample_index, quiet_mode)
    elif task == 'delta_caption':
        result = await evaluate_delta_caption_task(session, sample_data, debug_mode, sample_index, quiet_mode)
    elif task == 'shape_mating':
        result = await evaluate_shape_mating_task(session, sample_data, debug_mode, sample_index, quiet_mode)
    else:
        if not quiet_mode:
            print(f"Unknown task type: {task}")
        return {}, f"Unknown task: {task}", "Unknown task type", task
    
    # Return result with detected task type
    if result:
        metrics, response_text, reason = result
        return metrics, response_text, reason, task
    else:
        return {}, "No result", "No evaluation", task

async def evaluate_sample_async_with_retry(session, sample_data, debug_mode=False, sample_index=None, max_retries=3, quiet_mode=False):
    """Retry wrapper for evaluation"""
    for attempt in range(max_retries + 1):
        try:
            result = await evaluate_sample_async(session, sample_data, debug_mode, sample_index, quiet_mode)
            return result
        except Exception as e:
            if attempt < max_retries:
                import random
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            else:
                if not quiet_mode:
                    print(f"Failed after {max_retries} retries: {e}")
                return {}, f"Error: {e}", f"Error: {e}"
    
    return {}, "Unexpected error", "Unexpected error"

# ============================================================================
# Result Saving Functions
# ============================================================================

def save_metrics_only_to_json(results, output_file):
    """Save paper metrics by task type.

    Change Captioning reports B/R for verify and M for delta_caption.
    Delta-caption M is the raw GPT judge score on a 0-10 scale; M_percent is
    provided only as the normalized percentage for reporting convenience.
    """
    metrics_by_task = {}
    
    # Overall metrics
    overall_b_correct = 0
    overall_b_total = 0
    overall_r_correct = 0
    overall_r_total = 0
    overall_s_correct = 0
    overall_s_total = 0
    overall_m_correct = 0
    overall_m_total = 0
    
    for task_type, metrics in results.items():
        task_metrics = {}
        
        # Binary (B) - for verify
        if 'B' in metrics and metrics['B']:
            b_correct = sum(metrics['B'])
            b_total = len(metrics['B'])
            task_metrics['B'] = round(b_correct / b_total, 4) if b_total > 0 else None
            overall_b_correct += b_correct
            overall_b_total += b_total
        
        # Reasoning (R) - for verify and shape_mating.
        # R is reported on samples whose first-stage decision is correct.
        # Shape-mating wrong selections carry R=None; verify stores B/R for all
        # samples, so filter verify by B=1 here.
        if 'R' in metrics and metrics['R']:
            if task_type == 'verify' and 'B' in metrics and len(metrics['B']) == len(metrics['R']):
                r_values = [
                    r for b, r in zip(metrics['B'], metrics['R'])
                    if b == 1 and r is not None
                ]
            else:
                r_values = [r for r in metrics['R'] if r is not None]
            if r_values:
                r_correct = sum(r_values)
                r_total = len(r_values)
                task_metrics['R'] = round(r_correct / r_total, 4) if r_total > 0 else None
                task_metrics['R_count'] = r_total
                overall_r_correct += r_correct
                overall_r_total += r_total
        
        # Selection (S) - for shape_mating
        if 'S' in metrics and metrics['S']:
            s_correct = sum(metrics['S'])
            s_total = len(metrics['S'])
            task_metrics['S'] = round(s_correct / s_total, 4) if s_total > 0 else None
            overall_s_correct += s_correct
            overall_s_total += s_total
        
        # Semantic (M) - for delta_caption (raw 0-10 scale)
        if 'M' in metrics and metrics['M']:
            m_sum = sum(metrics['M'])
            m_total = len(metrics['M'])
            if m_total > 0:
                m_raw = m_sum / m_total
                task_metrics['M'] = round(m_raw, 2)
                task_metrics['M_raw_0_10'] = round(m_raw, 2)
                task_metrics['M_percent'] = round(m_raw / 10.0 * 100.0, 2)
            overall_m_correct += m_sum
            overall_m_total += m_total
        
        # Sample count (use S as primary count, fallback to other metrics)
        if 'S' in metrics and metrics['S']:
            total_samples = len(metrics['S'])
        elif 'B' in metrics and metrics['B']:
            total_samples = len(metrics['B'])
        else:
            total_samples = max((len(v) for v in metrics.values() if isinstance(v, list)), default=0)
        
        if total_samples > 0:
            task_metrics['count'] = total_samples
            metrics_by_task[task_type] = task_metrics
    
    # Overall metrics
    overall_metrics = {}
    if overall_b_total > 0:
        overall_metrics['B'] = round(overall_b_correct / overall_b_total, 4)
    if overall_r_total > 0:
        overall_metrics['R'] = round(overall_r_correct / overall_r_total, 4)
    if overall_s_total > 0:
        overall_metrics['S'] = round(overall_s_correct / overall_s_total, 4)
    if overall_m_total > 0:
        overall_m_raw = overall_m_correct / overall_m_total
        overall_metrics['M'] = round(overall_m_raw, 2)
        overall_metrics['M_raw_0_10'] = round(overall_m_raw, 2)
        overall_metrics['M_percent'] = round(overall_m_raw / 10.0 * 100.0, 2)
    
    # Final structure
    metrics_results = {
        'overall': overall_metrics,
        **metrics_by_task
    }
    
    # Save
    if output_file.endswith('_llm_eval.json'):
        metrics_output_file = output_file.replace('_llm_eval.json', '_metrics.json')
    else:
        root, ext = os.path.splitext(output_file)
        metrics_output_file = f"{root}_metrics{ext or '.json'}"
    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metrics saved to {metrics_output_file}")
    return metrics_results

# ============================================================================
# Main Function
# ============================================================================

async def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Change Captioning Evaluation Script")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent requests")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode: only show tqdm progress bar")
    parser.add_argument("--annotation", type=str, default=None, help="Path to annotation file with why_ref for reasoning evaluation")
    
    args = parser.parse_args()
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    quiet_mode = args.quiet
    
    # Resolve input file path
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return
    
    # Set output file path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Determine output directory based on input file path
        base_eval_dir = os.path.join("outputs", "llm_eval")
        
        if 'outputs' in input_file or 'evaluation' in input_file:
            # PointLLM/3DLLM results
            if 'shape_mating' in input_file.lower():
                output_dir = os.path.join(base_eval_dir, "3dllm", "shape_mating")
            elif 'change_captioning' in input_file.lower():
                output_dir = os.path.join(base_eval_dir, "3dllm", "change_captioning")
            else:
                output_dir = os.path.join(base_eval_dir, "3dllm")
        else:
            # VLM results
            if 'shape_mating' in input_file.lower():
                output_dir = os.path.join(base_eval_dir, "shape_mating")
            else:
                output_dir = os.path.join(base_eval_dir, "change_captioning")
        
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{base_name}_llm_eval.json")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
        # データ形式を判定して正規化
        if isinstance(json_data, list):
            # VLM形式（直接配列）
            data = json_data
        elif isinstance(json_data, dict) and 'results' in json_data:
            # PointLLM形式（results配列を持つ）
            data = json_data['results']
        else:
            print("Error: Unsupported JSON format. Expected a list or a dict with 'results' key.")
            return
        
        if args.max_samples:
            data = data[:args.max_samples]
    
    # Load annotation file for why_ref (ground truth reasoning)
    why_ref_map = {}
    if args.annotation:
        try:
            with open(args.annotation, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            for anno in anno_data:
                # Create key from object_ids
                obj_ids = tuple(anno.get('object_ids', []))
                if obj_ids:
                    why_ref_map[obj_ids] = anno.get('why_ref', {})
            if not quiet_mode:
                print(f"Loaded {len(why_ref_map)} annotations with why_ref from {args.annotation}")
        except Exception as e:
            print(f"Warning: Failed to load annotation file: {e}")
    
    # Add why_ref to each sample
    for sample in data:
        obj_ids = tuple(sample.get('object_ids', []))
        if obj_ids in why_ref_map:
            sample['why_ref'] = why_ref_map[obj_ids]
    
    if not quiet_mode:
        print(f"Starting Change Captioning evaluation of {input_file}...")
        print(f"Total samples: {len(data)}")
        print(f"Output: {args.output}")
    
    # Results storage
    results = {}
    detailed_results = []
    error_count = 0
    
    start_time = datetime.now()
    
    # Async batch processing
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pbar = tqdm(total=len(data), desc="Evaluating", unit="sample", disable=(not quiet_mode))
        
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            
            # Process batch
            tasks = [evaluate_sample_async_with_retry(session, sample, debug_mode=args.debug, sample_index=i + idx, quiet_mode=quiet_mode) 
                    for idx, sample in enumerate(batch)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for sample, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    error_count += 1
                    continue
                
                judgements, response_text, reason, task_type = result
                
                if not judgements:
                    error_count += 1
                    continue
                
                # Initialize task results
                if task_type not in results:
                    results[task_type] = {
                        'B': [],  # Binary (verify)
                        'R': [],  # Reasoning (verify, shape_mating)
                        'S': [],  # Selection (shape_mating)
                        'M': []   # Semantic (delta_caption)
                    }
                
                # Add scores
                for metric, score in judgements.items():
                    if score is not None:
                        results[task_type][metric].append(score)
                
                # Save detailed result
                detailed_results.append({
                    "sample_id": sample.get('sample_id', f'sample_{len(detailed_results)}'),
                    "task_type": task_type,
                    "judgements": judgements,
                    "question": sample.get('question', ''),
                    "ground_truth": sample.get('ground_truth', ''),
                    "model_answer": sample.get('prediction', sample.get('model_answer', '')),
                    "model_reasoning": sample.get('reasoning', ''),
                    "metadata": sample.get('metadata', {}),
                    "evaluation_response": response_text,
                    "reason": reason
                })
            
            # Update progress
            pbar.update(len(batch))
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        pbar.close()
    
    end_time = datetime.now()
    
    if not quiet_mode:
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        
        for task_type, metrics in results.items():
            print(f"\n{task_type}:")
            print("-" * 40)
            
            for metric, scores in metrics.items():
                if scores:
                    # M metric for delta_caption is 0-10 scale, show as average
                    if metric == 'M' and task_type == 'delta_caption':
                        avg_score = sum(scores) / len(scores)
                        print(
                            f"  {metric}: {avg_score:.2f}/10 "
                            f"({avg_score / 10.0 * 100.0:.2f}%, avg of {len(scores)} samples)"
                        )
                    elif metric == 'R' and task_type == 'verify' and metrics.get('B') and len(metrics['B']) == len(scores):
                        r_values = [
                            r for b, r in zip(metrics['B'], scores)
                            if b == 1 and r is not None
                        ]
                        if r_values:
                            accuracy = sum(r_values) / len(r_values) * 100
                            print(f"  {metric}: {accuracy:.2f}% ({sum(r_values)}/{len(r_values)})")
                    else:
                        # Other metrics are 0/1 binary, show as percentage
                        accuracy = sum(scores) / len(scores) * 100
                        print(f"  {metric}: {accuracy:.2f}% ({sum(scores)}/{len(scores)})")
        
        print(f"\n⏱️ Execution time: {(end_time - start_time).total_seconds():.1f} seconds")
        print(f"❌ Errors encountered: {error_count}")
    
    # Save detailed results
    output_data = {
        "evaluation_summary": {
            "input_file": input_file,
            "total_samples": len(data),
            "errors": error_count,
            "model_used": "gpt-4o-mini-2024-07-18",
            "evaluation_start_time": start_time.isoformat(),
            "evaluation_end_time": end_time.isoformat(),
            "execution_time_seconds": int((end_time - start_time).total_seconds())
        },
        "detailed_results": detailed_results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {args.output}")
    
    # Save metrics only
    save_metrics_only_to_json(results, args.output)

if __name__ == "__main__":
    asyncio.run(main())
