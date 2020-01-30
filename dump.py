import fasttext
import numpy as np
import joblib

def dump_split(sents_f, embed_model_f, model_f, prefix):
    model = joblib.load(model_f)
    embed_model = fasttext.load_model(embed_model_f)

    sentences = []
    embeddings = []

    with open(sents_f) as handle:
        for new_line in handle:
            if len(new_line.split()) < 5:
                continue
            sentences.append(new_line.strip())
            embeddings.append(embed_model.get_sentence_vector(new_line.strip()))

    preds = model.predict(embeddings)
    results = zip(preds, sentences)

    filenames = [prefix + str(i) + '.txt' for i in range(len(model.cluster_centers_))]
    handles = [open(f, 'w') for f in filenames]

    for pred, sent in results:
        handle = handles[pred]
        handle.write(sent)
        handle.write('\n')

    [h.close() for h in handles]

def dump_pred(sents_f, embed_model_f, model_f, dest):
    model = joblib.load(model_f)
    embed_model = fasttext.load_model(embed_model_f)

    sentences = []
    embeddings = []

    with open(sents_f) as handle:
        for new_line in handle:
            if len(new_line.split()) < 5:
                continue
            sentences.append(new_line.strip())
            embeddings.append(embed_model.get_sentence_vector(new_line.strip()))

    preds = model.predict(embeddings)
    results = zip(preds, sentences)

    with open(dest, 'w') as handle:
        for pred, sent in results:
            handle.write(str(pred))
            handle.write(' ')
            handle.write(sent)
            handle.write('\n')