# Author: oceaneLIU
# https://github.com/oceaneLIU/GraphCoder/blob/main/utils/metrics.py

import editdistance

def compute_EM(target, prediction, language="python"):
    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    if target_lines_str == prediction_lines_str:
        return 1
    else:
        return 0


def compute_ES(target, prediction, language="python"):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]

    target_str = ''.join(target_lines)
    prediction_str = ''.join(prediction_lines)
    ES_score = 1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    return ES_score