import pandas as pd
from rouge_score import rouge_scorer
import pandas as pd
import re
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
import time

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

    def parse_output(self):    

        # Split the text at the word "Response"
        parts = self.output.split("Response:", 1)

        # Check if "Response" is in the text and get the part after it
        if len(parts) > 1:
            response_text = parts[1].strip()
        else:
            response_text = ""

        return response_text