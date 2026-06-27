from torchmetrics.text import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from argparse import ArgumentParser
import json
import csv
import os
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--references', type=str, required=True, 
                        help='Path to JSON file containing reference texts (ground truth)')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to JSON file containing prediction texts')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to the output CSV file for appending results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load JSON files
    with open(args.references, 'r', encoding='utf-8') as f:
        references = json.load(f)
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Ensure both are lists
    if not isinstance(references, list):
        references = [references]
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    assert len(references) == len(predictions), \
        f"Number of references ({len(references)}) doesn't match predictions ({len(predictions)})"

    # Initialize Torchmetrics helpers
    bleu_helpers = {
        1: BLEUScore(n_gram=1, smooth=False),
        2: BLEUScore(n_gram=2, smooth=False),
        3: BLEUScore(n_gram=3, smooth=False),
        4: BLEUScore(n_gram=4, smooth=False),
    }
    bertscore_helper = BERTScore()
    rouge_helper = ROUGEScore()

    for reference, prediction in tqdm(zip(references, predictions), desc="Computing metrics", total=len(references)):
        if 'id' in reference and 'id' in prediction:
            assert reference['id'] == prediction['id'], \
                f"Reference and prediction IDs do not match: {reference['id']} vs {prediction['id']}"
        
        # 提取 'explanation' 字段
        ref_text = reference.get('explanation', '')
        pred_text = prediction.get('explanation', '')
        
        # 跳过空文本
        if not ref_text.strip() or not pred_text.strip():
            continue
        
        # Update Torchmetrics
        for n in [1, 2, 3, 4]:
            bleu_helpers[n].update([pred_text], [[ref_text]])
            
        bertscore_helper.update([pred_text], [[ref_text]])
        rouge_helper.update([pred_text], [[ref_text]])

    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print("="*50)
    
    # Compute BLEU
    bleu_results = {}
    for n in [1, 2, 3, 4]:
        bleu_score = bleu_helpers[n].compute().item()
        bleu_results[n] = bleu_score
        print(f"BLEU-{n}: {bleu_score:.4f}")
    
    # Compute BERTScore (取 F1 作为代表)
    bertscore_results = bertscore_helper.compute()
    bert_f1 = bertscore_results['f1'].mean().item()
    print(f"BERTScore F1: {bert_f1:.4f}")
    
    # Compute ROUGE-L (取 F-measure 作为代表)
    rouge_results = rouge_helper.compute()
    rouge_l_f1 = rouge_results['rougeL_fmeasure'].item()
    print(f"ROUGE-L F1: {rouge_l_f1:.4f}")
    print("="*50)

    # ==========================================
    # 追加写入 CSV 文件逻辑
    # ==========================================
    file_exists = os.path.isfile(args.output_csv)
    
    with open(args.output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 严格按照要求的顺序写入表头（剔除了 meteor）
        if not file_exists:
            writer.writerow(['input', 'bert-score', 'rouge-l', 'belu-1', 'belu-2', 'belu-3', 'belu-4'])
        
        # 写入本次运行的数据
        writer.writerow([
            args.predictions,
            f"{bert_f1:.4f}",
            f"{rouge_l_f1:.4f}",
            f"{bleu_results[1]:.4f}",
            f"{bleu_results[2]:.4f}",
            f"{bleu_results[3]:.4f}",
            f"{bleu_results[4]:.4f}"
        ])
    print(f"\nResults successfully appended to {args.output_csv}")

if __name__ == "__main__":
    main()