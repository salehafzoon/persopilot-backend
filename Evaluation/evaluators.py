from .UniEval.metric.evaluator import get_evaluator
from .UniEval.utils import convert_to_json


def calculate_unieval_scores(prompt, response):
    """
    Calculates UniEval scores for LLM response against user prompt.

    Args:
        prompt (str or list): User prompt(s)/question(s).
        response (str or list): LLM generated response(s) to be evaluated.

    Returns:
        list: A list of dictionaries containing UniEval scores.
    """
    # Convert to lists if single strings
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(response, str):
        response = [response]

    # Prepare inputs for UniEval (using prompt as context)
    data = convert_to_json(output_list=response, src_list=prompt, context_list=prompt)

    # Initialize the evaluator for dialogue tasks
    evaluator = get_evaluator('dialogue')

    # Evaluate and obtain scores
    eval_scores = evaluator.evaluate(data, print_result=False)
    
    # Round all scores to 3 decimal places
    for score_dict in eval_scores:
        for key, value in score_dict.items():
            score_dict[key] = round(float(value), 3)
            
    return eval_scores
