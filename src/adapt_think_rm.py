"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""

from math_verify import parse, verify

from .verify.evaluations import is_equiv
from .verify.utils import (
    _is_frac,
    _normalize,
    _str_is_int,
    are_equal_under_sympy,
    last_boxed_only_string,
    mathd_normalize_answer,
    remove_boxed,
    split_tuple,
)

# logging.info("DeepscaleR Here!!!")


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_hf_math(given_answer: str, ground_truth: str) -> bool:
    preds = parse(given_answer)
    gold = parse(ground_truth)

    return verify(gold, preds)


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    # elif grade_answer_sympy(given_answer, ground_truth):
    #     return True
    elif grade_answer_hf_math(given_answer, ground_truth):
        return True
    elif is_equiv(ground_truth, given_answer):
        return True
    return False


def _get_deepscaler_rule_base_reward(model_answer, label):
    if model_answer is None:
        return 0
    if label == "" or label is None:
        return 0

    model_answer = str(model_answer).strip()

    # Normalize label(s)
    if isinstance(label, (list, tuple)):
        ground_truths = [str(x).strip() for x in label]
    elif isinstance(label, (str, float, int)):
        ground_truths = [str(label).strip()]
    else:
        print(f"ERROR GROUND TRUTH: {label}")
        return 0

    # Process each ground truth (boxed, etc.)
    processed_ground_truths = []
    for truth in ground_truths:
        if "\\boxed" in truth:
            processed = extract_answer(truth)
            processed_ground_truths.append(processed or truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        try:
            if grade_answer_mathd(model_answer, ground_truth):
                return 1
        except Exception:
            # defensive: do not crash reward
            continue

    return 0


def adapt_think_rm(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """

    # logging.info("check format!!!")
    if "<think>" in solution_str:
        pred = "ERROR: Multiple <think>"
        # print(pred)
        acc = 0
    elif solution_str.count("</think>") != 1:
        pred = f"ERROR: Num of </think> == {solution_str.count('</think>')}"
        acc = 0
    else:
        model_solution = solution_str.split("</think>")[-1]
        pred = extract_answer(model_solution)
        acc = _get_deepscaler_rule_base_reward(pred, ground_truth)

        if pred is None:
            pred = "ERROR: Answer Extraction Failed"
            # print(pred)

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }


def nothinking_rm(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """

    # logging.info("check format!!!")
    if "<think>" in solution_str:
        pred = "ERROR: Multiple <think>"
        acc = 0
    # elif ('</think>') in solution_str:
    #     pred = f"ERROR: Num of </think> == {solution_str.count('</think>')}"
    #     acc = 0
    else:
        model_solution = solution_str.strip()
        pred = extract_answer(model_solution)
        acc = _get_deepscaler_rule_base_reward(pred, ground_truth)

        if pred is None:
            pred = "ERROR: Answer Extraction Failed"

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }


def multi_choice_rm(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    pred = extract_answer(solution_str)

    if pred is None:
        pred = "ERROR: Answer Extraction Failed"
        acc = 0
    else:
        pred = pred.strip()
        ground_truth = ground_truth.strip()
        acc = 1.0 if pred == ground_truth else 0.0

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }


def hf_math_rm(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """

    model_solution = solution_str.strip()[-500:]

    acc = grade_answer_hf_math(model_solution, ground_truth)

    if preds is None or preds == []:
        pred = "ERROR: Answer Extraction Failed"
        # print(pred)
    else:
        pred = str(preds[0])
    assert isinstance(pred, str), preds

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }


if __name__ == "__main__":
    solution = "xxxx"
    print(hf_math_rm("", solution, "1"))
