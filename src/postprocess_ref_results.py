import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import orjson
import hashlib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from adapt_think_rm import adapt_think_rm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--input_path", type=str, default="./data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_n10_K16_len16384.jsonl")
    parser.add_argument("--output_dir", type=str, default="./data/train/ref_results/")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--nothinking", action='store_true', default=False)
    return parser.parse_args()


def process(problem):
    """문제에 대한 여러 솔루션 시도를 분석하고 성능 지표를 계산"""
    items = data.get(problem)

    real_answer = items[0].get(args.answer_key)

    solutions = []
    lengths = []
    truncates = []

    prefix = '' if args.nothinking else '</think>'
    token_adjustment = 0 if args.nothinking else 1

    for item in items:
        choice = item['response']['choices'][0]
        usage = item['response']['usage']

        solutions.append(prefix + choice['text'])
        lengths.append(usage['completion_tokens'] + token_adjustment)
        truncates.append(choice['finish_reason'] == 'length')

    # 정확도 계산
    correctness = [
        adapt_think_rm(data_source='', solution_str=sol, ground_truth=real_answer).get('acc')
        for sol in solutions
    ]

    # 문제 텍스트 정규화
    processed_problem = tokenizer.decode(
        tokenizer.encode(problem, add_special_tokens=False)
    )

    return {
        'problem': processed_problem.strip(),
        'answer': real_answer,
        'metrics': {
            'n_responses': len(items),
            'avg_acc_thinking': np.mean(correctness),
            'avg_len_thinking': np.mean(lengths),
            'avg_clip_ratio': np.mean(truncates),
        }
    }


if __name__ == "__main__":
    # 초기화
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 데이터 로드
    data = defaultdict(list)
    with open(args.input_path, 'rb') as f:
        for line in tqdm(f):
            item = orjson.loads(line)
            problem = item["problem"]
            key = hashlib.md5(problem.encode()).hexdigest()
            data[key].append(item)

    print(f"총 문제 수: {len(data)}")

    # 병렬 처리
    problems = list(data.keys())
    with Pool(64) as pool:
        results = list(tqdm(pool.imap(process, problems), total=len(problems)))

    # 전체 메트릭 계산
    metric_keys = results[0]['metrics'].keys()
    overall_metrics = {
        key: np.mean([r['metrics'][key] for r in results])
        for key in metric_keys
    }

    # 전체 메트릭을 결과 앞에 추가
    results.insert(0, {
        'problem': '__OVERALL__',
        'answer': None,
        'metrics': overall_metrics
    })

    # 메트릭 출력
    for key, value in overall_metrics.items():
        print(f'{key}: {value}')

    # 결과 저장
    output_dir = os.path.dirname(args.output_dir)
    file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    output_path = os.path.join(
        output_dir, f"{file_name}.json"
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
