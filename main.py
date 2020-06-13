# %%
import torch
import string

from transformers import BertTokenizer, BertForMaskedLM

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
# bert_model = BertForMaskedLM.from_pretrained('bert-base-german-cased').eval()

bert_tokenizer = BertTokenizer.from_pretrained(
    '/home/smanjil/Downloads/frozen/')
bert_model = BertForMaskedLM.from_pretrained(
    '/home/smanjil/Downloads/frozen/').eval()


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = list()

    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token)
    print('##kos' in tokens[:top_clean], tokens[:top_clean])

    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = tokenizer.tokenize(text_sentence)
    print(f"Tokenized sentence: {text_sentence}")

    # mask
    text_sentence[8] = tokenizer.mask_token
    text_sentence[2] = tokenizer.mask_token
    print(f"Masked sentence: {text_sentence}")

    if tokenizer.mask_token == text_sentence[-1]:
        text_sentence.append('.')

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    print(f"After encoding: {[tokenizer.decode(j.item()) for j in input_ids[0]]}")

    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    print(f"\nOriginal sentence: {text_sentence}")
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)

    with torch.no_grad():
        predict = bert_model(input_ids)[0]

    first_mask = decode(bert_tokenizer, predict[0, mask_idx, :].topk(
        top_clean).indices.tolist(), top_clean)

    second_mask = decode(bert_tokenizer, predict[0, 9, :].topk(
        top_clean).indices.tolist(), top_clean)

    return {'first': first_mask,
            'second': second_mask}
