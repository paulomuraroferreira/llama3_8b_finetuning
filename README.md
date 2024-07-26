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

Then, we define the Trainer class and the training configuration:


```python
class Trainer:
    def __init__(self, model, tokenizer, train_dataset, peft_config, use_flash_attention2, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.peft_config = peft_config
        self.args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=use_flash_attention2,
            fp16=not use_flash_attention2,
            tf32=use_flash_attention2,
            max_grad_norm=0.3,
            warmup_steps=5,
            lr_scheduler_type="linear",
            disable_tqdm=False,
            report_to="none"
        )
        self.model = get_peft_model(self.model, self.peft_config)

    def train_model(self, format_instruction_func):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            peft_config=self.peft_config,
            max_seq_length=2048,
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=format_instruction_func, 
            args=self.args,
        )
        trainer.train()
        return trainer
```

Finally, the classes are called and the training starts.

Note that the Llama models are gated, i.e., HuggingFace requires a token given after   
the terms of the model are accepted and Meta approves the access (which is almost instantly). 


```python
dataset_handler = DatasetHandler(data_path=utils.Variables.INSTRUCTION_DATASET_JSON_PATH)
train_dataset, test_dataset = dataset_handler.load_and_split_dataset()

new_test_dataset = []
for dict_ in test_dataset:
    dict_['Output'] = ''
    new_test_dataset.append(dict_)

model_manager = ModelManager(
    model_id="meta-llama/Meta-Llama-3-8B",
    use_flash_attention2=True,
    hf_token=os.environ["HF_TOKEN"]
)
model, tokenizer = model_manager.load_model_and_tokenizer()
model_manager.save_model_and_tokenizer(model, tokenizer, save_directory=utils.Variables.BASE_MODEL_PATH)
model = model_manager.prepare_for_training(model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    peft_config=peft_config,
    use_flash_attention2=True,
    output_dir=utils.Variables.FINE_TUNED_MODEL_PATH
)
trained_model = trainer.train_model(format_instruction_func=dataset_handler.format_instruction)
trained_model.save_model()
```

# Step 4 - Evaluation

In order to evaluate the fine tuning results, we employed the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Score, which 
compares the overlap between two sets of text, in order to measure similarity between them.

Specifically, we employ the rouge_scorer library to calculate Rouge1 and Rouge2, which measures the 1-gram and 2-gram overlap between the texts.


```python
import pandas as pd
from rouge_score import rouge_scorer

def calculate_rouge_scores(generated_answers, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    for gen, ref in zip(generated_answers, ground_truth):
        scores = scorer.score(gen, ref)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
    average_rouge1 = total_rouge1 / len(generated_answers)
    average_rouge2 = total_rouge2 / len(generated_answers)
    average_rougeL = total_rougeL / len(generated_answers)
    return {'average_rouge1':average_rouge1,
            'average_rouge2':average_rouge2,
            'average_rougeL':average_rougeL}
```

In order to perform this calculation, we take the instructions from the test dataset, pass them into both the base model  
and the fine-tuned model, and compare it to the expected output from the instruction/answer dataset. 

The code for the evaluation can be found at `/llama3_8b_finetuning/model_evaluation.py`.


```python
class ModelHandler:

    def __init__(self):
        pass

    def loading_model(self, model_chosen='fine_tuned_model'):

        if model_chosen == 'fine_tuned_model':
            model_dir=utils.Variables.FINE_TUNED_MODEL_PATH
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                )

        elif model_chosen == 'base_model':
            model_dir=utils.Variables.BASE_MODEL_PATH
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def ask_question(self, instruction, temperature=0.5, max_new_tokens = 1000):

        prompt = format_instruction(instruction)

        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(input_ids=input_ids, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.5,temperature=temperature)
        end_time = time.time()

        total_time = end_time - start_time
        output_length = len(outputs[0])-len(input_ids[0])

        self.output = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

        return self.output
```

# Evaluation Results

The rouge scores are as follows:
    
FINE-TUNED MODEL:

`{'average_rouge1': 0.39997816307812206,
 'average_rouge2': 0.2213826792342886,
 'average_rougeL': 0.33508922374837047}`

BASE MODEL: 

`{'average_rouge1': 0.2524191394349585,
 'average_rouge2': 0.13402054342344535,
 'average_rougeL': 0.2115590931984475}`  
   
     
     
Therefore, it can be seen that the perfomance for the fine-tuned model on the test dataset are 
significatly superior than the base model.

# Alternatives

It tooks quite a while to write this code and get it working. It was good to practice, but for everyday fine-tuning related    
jobs, just use hugging face autotrainer hosted locally (https://github.com/huggingface/autotrain-advanced): 
