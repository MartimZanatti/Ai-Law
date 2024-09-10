import os

import torch
from emb_model import load_bert_model, load_word_2_vec
import shutil
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt



def calculate_embeddings(doc, model_name_bert="stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1", sections=["fundamentação de direito"], device="cpu"):
    considered_sections_text = []
    considered_sections_ids = []
    for p in doc.paragraphs:
        if p.zone in sections: #if the paragraph belong to the considered sections
            considered_sections_text.append(p.text.get_text())
            considered_sections_ids.append(p.id)


    if device == 'cuda':
        torch.cuda.set_device()

    model_emb = load_bert_model(model_name_bert, device)

    X = torch.zeros((len(considered_sections_text), 1024), device=device)

    for i,p in enumerate(considered_sections_text):
        emb = torch.from_numpy(model_emb.encode(p))  # -> (1024) -> X[i,j]

        if emb.sum().data == 0:
            print("unkown paragraph", p)
            X[i] = torch.from_numpy(model_emb.encode("UNK"))

        else:
            X[i] = emb

    return X, considered_sections_ids



def calculate_embeddings_neural_shift(doc, model_name_bert="stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1", sections=["fundamentação"], device="cpu"):
    considered_sections_text = []
    considered_sections_ids = []
    for p in doc.paragraphs:
        if p.zone in sections: #if the paragraph belong to the considered sections
            considered_sections_text.append(p.text)
            considered_sections_ids.append(p.id)

    if device == 'cuda':
        torch.cuda.set_device()


    model_emb = load_bert_model(model_name_bert, device)

    X = torch.zeros((len(considered_sections_text), 1024), device=device)

    for i,p in enumerate(considered_sections_text):
        emb = torch.from_numpy(model_emb.encode(p))  # -> (1024) -> X[i,j]

        if emb.sum().data == 0:
            print("unkown paragraph", p)
            X[i] = torch.from_numpy(model_emb.encode("UNK"))

        else:
            X[i] = emb

    return X, considered_sections_ids



def calculate_embeddings_word2vec(doc, model_name_word2vec="word2vec.model", sections=["fundamentação"], device="cpu"):
    considered_sections_text = []
    considered_sections_ids = []
    for p in doc.paragraphs:
        if p.zone in sections: #if the paragraph belong to the considered sections
            considered_sections_text.append(p.text.get_text())
            considered_sections_ids.append(p.id)

    if device == 'cuda':
        torch.cuda.set_device()


    model_emb = load_word_2_vec(model_name_word2vec)

    X = torch.zeros((len(considered_sections_text), 100), device=device)


    for i,p in enumerate(considered_sections_text):
        words = p.split()
        num_words = 0
        p_emb = torch.zeros((100),device=device)
        for w in words:
            try:
                w_emb = model_emb.wv[w]
                w_emb_copy = np.copy(w_emb)
                w_emb_torch = torch.FloatTensor(w_emb_copy)
                p_emb += p_emb + w_emb_torch
                num_words += 1
            except:
                continue

        if num_words != 0:
            p_emb = torch.div(p_emb, num_words)

        X[i] = p_emb

    return X, considered_sections_ids



def find_longest_section(denotations):
    longest_sequence = 0
    section = ""

    for d in denotations:
        if d["type"] in ["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "foot-note"]:
            length = d["end"] - d["start"] + 1
            if length > longest_sequence:
                longest_sequence = length
                section = d["type"]
        else:
            length = len(d["zones"])
            if length > longest_sequence:
                longest_sequence = length
                section = d["type"]

    return section


def check_sections_to_summarize(output, sections, sections_returned):
    section_added = None
    denotations = output["denotations"]

    found_sections = []

    for d in denotations:
        found_sections.append(d["type"])

    intersection_sections = [value for value in sections if value in found_sections]

    sections_returned_found = [value for value in sections_returned if value in intersection_sections]

    if intersection_sections == [] or sections_returned_found == []:
        section = find_longest_section(denotations)
        sections.append(section)
        section_added = section
        return sections, section_added
    else:
        return sections, section_added


def transform_output_zones(output):
    new_output = {"wrappe": output["wrapper"], "text": output["text"], "denotations": []}
    text = output["text"]
    denotations = output["denotations"]
    new_denotations = []

    for d in denotations:
        if d["type"] in ["colectivo", "declaração"]:
            zones = []
            lst_zones = list(range(d["start"], d["end"] + 1))
            for z in lst_zones:
                zones.append((z,))
            zone_dict = {"id": d["id"], "zones": zones, "type": d["type"]}
            new_denotations.append(zone_dict)
        elif d["type"] == "título":
            titulos_ids = d["titulos_ids"]
            zones = []
            for t in titulos_ids:
                zones.append((t,))
            zone_dict = {"id": d["id"], "zones": zones, "type": d["type"]}
            new_denotations.append(zone_dict)
        else:
            new_denotations.append(d)

    new_output["denotations"] = new_denotations

    return new_output


def zip_dir():
    shutil.make_archive("results", 'zip', "./results/")




def separate_dataset():
    path = "../IrisDataset/zones_dataset/"
    new_path = "../IrisDataset/zones_dataset_original/"
    files = os.listdir(path)
    print(files)
    for f in files:
        if "no_cabecalho" not in f and "no_title" not in f and "no_colectivo" not in f:
            shutil.move(path + f, new_path + f)


def get_zones(path, file_name): #for neural shift example
    dict_text = {}
    previous_normalized_type = "undifined"
    f = open(path + file_name, encoding="UTF-8")
    data = json.load(f)
    for key,value in data.items():
        if previous_normalized_type == "undifined":
            if key == "text":
                previous_value = value
            if key == "children":
                if value != []:
                    if "normalized_type" in value[0]:
                        if value[0]["normalized_type"] in ["Fundamentação"]:
                            previous_normalized_type = "Fundamentação"
                        elif value[0]["normalized_type"] in ["Factos", "Factos provados"]:
                            previous_normalized_type = "Fundamentação de Facto"
                        elif value[0]["normalized_type"] in ["Direito", "Objecto", "Apreciando/Decidir"]:
                            previous_normalized_type = "Fundamentação de Direito"
                        elif value[0]["normalized_type"] in ["Dispositivo/Decisão", "Conclusão"]:
                            previous_normalized_type = "Decisão"
                        else:
                            previous_normalized_type = value[0]["normalized_type"]
                    else:
                        previous_normalized_type = "Fundamentação"
                else:
                    previous_normalized_type = "Fundamentação"
                    text = assing_text(previous_value[1:], [previous_value[0]])
                    dict_text[previous_normalized_type] = text

                if value != []:
                    dict_text, text, previous_normalized_type = deal_children(value, [], previous_normalized_type, dict_text)

                    dict_text[previous_normalized_type] = text
    return dict_text



def deal_children(children, text, previous_normalized_type, dict_text): #for neural shift example
    for c in children:
        if "normalized_type" in c:
            if c["normalized_type"] != previous_normalized_type:
                if c["normalized_type"] in ["Fundamentação"]:
                    if "Fundamentação" not in dict_text:
                        if previous_normalized_type != "Fundamentação":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Fundamentação"
                            text = []

                elif c["normalized_type"] in ["Factos", "Factos provados"]:
                    if "Fundamentação de Facto" not in dict_text:
                        if previous_normalized_type != "Fundamentação de Facto":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Fundamentação de Facto"
                            text = []
                    else:
                        if previous_normalized_type != "Fundamentação de Facto":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Fundamentação de Facto"
                            text = dict_text["Fundamentação de Facto"]

                elif c["normalized_type"] in ["Direito", "Objecto", "Apreciando/Decidir"]:
                    if "Fundamentação de Direito" not in dict_text:
                        if previous_normalized_type != "Fundamentação de Direito":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Fundamentação de Direito"
                    else:
                        if previous_normalized_type != "Fundamentação de Direito":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Fundamentação de Direito"
                            text = dict_text["Fundamentação de Direito"]

                elif c["normalized_type"] in ["Dispositivo/Decisão", "Conclusão"]:
                    if "Decisão" not in dict_text:
                        if previous_normalized_type != "Decisão":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Decisão"
                            text = []
                    else:
                        if previous_normalized_type != "Decisão":
                            dict_text[previous_normalized_type] = text
                            previous_normalized_type = "Decisão"
                            text = dict_text["Decisão"]
                else:
                    dict_text[previous_normalized_type] = text
                    previous_normalized_type = c["normalized_type"]
                    text = []


        current_text = c["text"]

        text = assing_text(current_text, text)
        if c["children"] != []:
           dict_text, text, previous_normalized_type = deal_children(c["children"], text, previous_normalized_type, dict_text)


    return dict_text, text, previous_normalized_type


def assing_text(current_text, text): #for neural shift example
    for t in current_text:
        if isinstance(t, str):
                text.append(t)
        else:
            children_text = t["children"]
            text = assing_text(children_text, text)

    return text


def create_panda_llm_rouge():
    llm_path = "results/grades-gpt4.csv"
    df = pd.read_csv(llm_path)
    print(df)
    scores_path = "results/stjirisbert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-v0-no-ground-truth/2-fundamentacao-relatório-fundamentacao-no-italic/rouge_scores/"
    score_files = os.listdir(scores_path)
    #print(score_files)
    recalls = []

    for i, row in enumerate(df.iloc):
        file_name = row["filename"][:-5]
        for f in score_files:
            if file_name == f[:-4]:
                file = open(scores_path + f, "r")
                lines = file.read().split('\n')
                r = float(lines[3][3:])
                recalls.append(r)
                break


    df["ROUGE-1"] = recalls

    df = df.replace(to_replace=-1,
               value=np.nan)

    df = df.dropna(subset=['Completeness', 'ROUGE-1'])
    df = df.reset_index(drop=True)


    accuracy_scores = df['Completeness']

    print(sum(accuracy_scores)/len(accuracy_scores))


    rouge1_scores = df['ROUGE-1']

    plt.figure(figsize=(10, 6))
    plt.scatter(accuracy_scores, rouge1_scores, alpha=0.5)
    plt.title('Coherence vs ROUGE-1')
    plt.xlabel('Coherence Scores')
    plt.ylabel('ROUGE-1 Scores')
    plt.grid(True)
    plt.show()











