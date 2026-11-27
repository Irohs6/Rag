from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging


class Pipeline:
    MODEL_NAME = "Qwen/Qwen3-0.6B"

    @staticmethod
    def init_model():
        logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(Pipeline.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(Pipeline.MODEL_NAME)
        model.to("cpu")
        model.eval()

        return tokenizer, model
