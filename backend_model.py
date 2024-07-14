import pandas as pd
import re
from peft import AutoPeftModelForCausalLM
import time
import utils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

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

    @staticmethod
    def format_instruction(sample):
        return f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {sample['Instruction']}

        ### Input:
        {sample['Input']}

        ### Response:
        {sample['Output']}
        """

    def ask_question(self, instruction, temperature=0.5, max_new_tokens = 1000):

        prompt = self.format_instruction(instruction)

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