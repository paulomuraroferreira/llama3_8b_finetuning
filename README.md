# README

This is an end-to-end project including data ingestion, creation of instruction/answer pairs, fine-tuning, and evaluation of the results.

# First Step - Data Scraping

Start by installing the dependencies with 

`pip install -r requirements.txt`

In order to find data for the fine-tuning, Arxiv was scraped for LLM papers published after the Llama 3 release date.

The Selenium scraping code can be found in `llama3_8b_finetuning/arxiv_scraping/Arxiv_pdfs_download.py`. (The webdriver must be downloaded before this script execution).

The scraping code takes the papers on the first Arxiv page and downloads them into the `llama3_8b_finetuning/data/pdfs` folder.

# Second Step - Instructions/Answer Pairs Creation

The code for this step can be found at `/llama3_8b_finetuning/creating_instruction_dataset.py`

The text content from the downloaded papers was parsed using Langchain's `PyPDFLoader`. Then, the text was sent into the Llama 3 70B model via Grok. Grok was chosen due to its speed and small cost. Also, it must be noted that the Llama 3 user license only allows its use for training/fine tuning Llama LLMs. Therefore, we wouldn't be able to use Llama 3 for create instructions/answer pairs for other models, even open-source ones or for non-commercial use.

The prompt for the pairs creation are on the utils file, and it can also be seen below: 

'''

    You are a highly intelligent and knowledgeable assistant tasked with generating triples of instruction, input, and output from academic papers related to Large Language Models (LLMs). Each triple should consist of:

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
    Output:
'''


Finally, the instructions are saved on `llama3_8b_finetuning/data/arvix_instruction_dataset.json`.


# Third Step - Fine Tuning

The code for this step can be found on `/llama3_8b_finetuning/model_trainer.py`


First we load the instructions/answer pairs, split them into test and train dataset, and format 
into the right format.


```python
class DatasetHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_and_split_dataset(self):
        dataset = load_dataset("json", data_files=self.data_path)
        train_test_split = dataset['train'].train_test_split(test_size=0.2)
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })
        return dataset_dict['train'], dataset_dict['test']

    @staticmethod
    def format_instruction(sample):
        return f"""
        Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.

        ### Instruction:
        {sample['Instruction']}

        ### Input:
        {sample['Input']}

        ### Response:
        {sample['Output']}
        """
```

Then, we define the class that loads the model and tokenizer from huggingface. 


```python
class ModelManager:
    def __init__(self, model_id, use_flash_attention2, hf_token):
        self.model_id = model_id
        self.use_flash_attention2 = use_flash_attention2
        self.hf_token = hf_token
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention2 else torch.float16
        )
    
    def load_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            quantization_config=self.bnb_config, 
            use_cache=False, 
            device_map="auto",
            token=self.hf_token,  
            attn_implementation="flash_attention_2" if self.use_flash_attention2 else "sdpa"
        )
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return model, tokenizer
```
