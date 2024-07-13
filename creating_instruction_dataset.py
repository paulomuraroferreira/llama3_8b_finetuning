from langchain_community.document_loaders.pdf import PyPDFLoader
import glob
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re
import json
import utils
import os
from utils import Variables

class FineTuning:

    def __init__(self):
        self.folder_path = Variables.PDFS_PATH
        self.pdf_files = glob.glob(os.path.join(self.folder_path, '*'))

        chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            max_retries = 30
        )

        system = Variables.INSTRUCTION_FOR_CREATING_TRIPLES

        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        self.chain = prompt | chat

    def loading_pdfs(self):

        self.loaded_pdfs = [PyPDFLoader(pdf).load() for pdf in self.pdf_files]

    def creating_instruction_dataset(self):

        self.list_instructions_unstructured = []
        self.list_pages_processed = []

        for pdf_paper in self.loaded_pdfs:
            for pdf_page in pdf_paper:
                try:
                    answer = self.chain.invoke({"text": f"{pdf_page.page_content}"})
                    self.list_instructions_unstructured.append(answer)
                    self.list_pages_processed.append((pdf_paper, pdf_page))
                    print(f'Page {pdf_page.metadata['page']} from paper {pdf_page.metadata['source']} processed.')

                except Exception as e:
                    print(f'Error on paper {pdf_page.metadata['source']} on page {pdf_page.metadata['page']}: {str(e)}.')
                    continue

    @staticmethod
    def formating_instruction_dataset(text):

        instruction_pattern = re.compile(r'Instruction: (.*?)\n', re.DOTALL)
        input_pattern = re.compile(r'Input: (.*?)\n', re.DOTALL)
        output_pattern = re.compile(r'Output: (.*?)(?=Triple|\Z)', re.DOTALL)

        # Extract matches
        instructions = instruction_pattern.findall(text)
        inputs = input_pattern.findall(text)
        outputs = output_pattern.findall(text)

        # Create a list of dictionaries
        triples = []
        for i in range(len(instructions)):
            triple = {
                "Instruction": instructions[i],
                "Input": inputs[i],
                "Output": outputs[i]
            }
            triples.append(triple)

        return triples
    
    def return_instruction_dataset(self):
        self.loading_pdfs()
        self.creating_instruction_dataset()

        list_triples = []

        for instruction_element in self.list_instructions_unstructured:
            triple = self.formating_instruction_dataset(instruction_element.content)
            list_triples.append(triple)

        self.list_triples = list_triples

        list_of_dicionaries = []
        for triple_ in self.list_triples:
            for instruction_set in triple_:
                list_of_dicionaries.append(instruction_set)

        with open(Variables.INSTRUCTION_DATASET_JSON_PATH, "w") as f:
            json.dump(list_of_dicionaries, f)

        self.list_of_dicionaries = list_of_dicionaries
        return self.list_of_dicionaries

if __name__ == '__main__':
    finetuning = FineTuning()
    triples = finetuning.return_instruction_dataset()
