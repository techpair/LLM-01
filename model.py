from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "roberta-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
