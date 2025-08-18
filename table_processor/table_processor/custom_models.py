from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch


class CustomModelForSequenceClassification:

    def __init__(self, model_pth, model_name):
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_pth)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.key_words = ["select", 'find', 'compute', 'calculate', 'determine', 'predict', 'estimate', 'infer', 'derive', 'obtain', 'get', 'retrieve', 'extract', 'identify', 'recognize', 'diagnose', 'detect', 'classify', 'categorize', 'label', 'tag', 'assign', 'organize', 'arrange', 'sort', 'group', 'cluster', 'combine', 'merge', 'unite', 'join', 'link', 'connect',
                          'associate', 'relate', 'compare', 'contrast', 'differentiate', 'distinguish', 'discriminate', 'separate', 'partition', 'split', 'divide', 'break', 'analyze', 'examine', 'inspect', 'explore', 'investigate', 'study', 'scrutin', 'list', 'top', 'lowest', 'highest', 'item', 'row', 'column', 'order', 'correlation', 'by', 'plot', 'histogram', 'scatter', 'bar']
        self.anti_key_words = ['how to', 'summarize', 'summary']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(input_ids=inputs.input_ids,
                             attention_mask=inputs.attention_mask).logits
        p = torch.nn.functional.softmax(outputs, dim=1)
        p[0] = p[0] + 0.41
        class_idx = p.argmax().item()
        q = text.lower()
        for kw in self.key_words:
            if kw in q:
                return 0
        for kw in self.anti_key_words:
            if kw in q:
                return 1

        return class_idx
