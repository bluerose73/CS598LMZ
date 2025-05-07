from argparse import ArgumentParser
import json
from fid.eval.metrics import compute_EM, compute_ES

parser = ArgumentParser()
parser.add_argument('--completion-path', type=str, required=True,
                    help="Path to the completion jsonl file.")
parser.add_argument('--language', type=str, required=True,
                    help="Programming language of the completion data.")
args = parser.parse_args()


EM_list = []
ES_list = []

def postprocess_line_completion(completion: str, language: str) -> str:
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"
    
    completion = completion.strip()
    lines = completion.split('\n')
    for line in lines:
        line = line.strip()
        if not line.startswith(comment_prefix):
            return line
    return lines[0]

with open(args.completion_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gt = data['ground_truth']
        completion = data['completion']
        completion = postprocess_line_completion(completion, args.language)
        EM = compute_EM(gt, completion, args.language)
        ES = compute_ES(gt, completion, args.language)
        EM_list.append(EM)
        ES_list.append(ES)

print(f"EM: {sum(EM_list) / len(EM_list)}")
print(f"ES: {sum(ES_list) / len(ES_list)}")