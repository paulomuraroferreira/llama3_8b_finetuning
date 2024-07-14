from pathlib import Path
import os

from dataclasses import dataclass
@dataclass
class Variables:
    PDFS_PATH:str = str(Path(os.path.abspath(__file__)).parent / 'data' / 'pdfs')
    INSTRUCTION_DATASET_JSON_PATH:str = str(Path(os.path.abspath(__file__)).parent / 'data' / 'arvix_instruction_dataset.json')
    BASE_MODEL_PATH:str = str(Path(os.path.abspath(__file__)).parent / 'models' / 'original_llama3_model')
    FINE_TUNED_MODEL_PATH:str = str(Path(os.path.abspath(__file__)).parent / 'models' / 'fine_tuned_model')
    INSTRUCTION_FOR_CREATING_TRIPLES = '''You are a highly intelligent and knowledgeable assistant tasked with generating triples of instruction, input, and output from academic papers related to Large Language Models (LLMs). Each triple should consist of:

    Instruction: A clear and concise task description that can be performed by an LLM.
    Input: A sample input that corresponds to the instruction.
    Output: The expected result or answer when the LLM processes the input according to the instruction.
    Below are some example triples:

    Example 1:

    Instruction: Summarize the following abstract.
    Input: "In this paper, we present a new approach to training large language models by incorporating a multi-task learning framework. Our method improves the performance on a variety of downstream tasks."
    Output: "A new multi-task learning framework improves the performance of large language models on various tasks."
    Example 2:

    Instruction: Provide a brief explanation of the benefits of using multi-task learning for large language models.
    Input: "Multi-task learning allows a model to learn from multiple related tasks simultaneously, which can lead to better generalization and performance improvements across all tasks. This approach leverages shared representations and can reduce overfitting."
    Output: "Multi-task learning helps large language models generalize better and improve performance by learning from multiple related tasks simultaneously."
    Now, generate similar triples based on the provided text from academic papers related to LLMs:

    Source Text
    (Provide the text from the academic papers here)

    Generated Triples
    Triple 1:

    Instruction:
    Input:
    Output:
    Triple 2:

    Instruction:
    Input:
    Output:
    Triple 3:

    Instruction:
    Input:
    Output:'''



