from transformers import pipeline
from ftlangdetect import detect

def detox_text(response):
    # detect language
    result = detect(text=response, low_memory=False)
    lang = result['lang']
    verdict = None
    if lang == 'en':
        clf = pipeline("text-classification", model="s-nlp/roberta_first_toxicity_classifier")
        verdict = clf(response)[0]['label']
    elif lang == 'ru':
        clf = pipeline("text-classification", model="s-nlp/russian_toxicity_classifier")
        verdict = clf(response)[0]['label']
    if verdict == 'toxic' or verdict == 'LABEL_1':
        if lang == 'en':
            pipe = pipeline("text2text-generation", model="s-nlp/bart-base-detox")
            return pipe(response)[0]['generated_text']
        elif lang == 'ru':
            pipe = pipeline("text2text-generation", model="s-nlp/ruT5-base-detox")
            return pipe(response)[0]['generated_text']
    return response