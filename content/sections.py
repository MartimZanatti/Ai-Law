from bilstm_utils import id2word


section_types = {
    'cabeçalho': 'Cabeçalho',
    'relatório': 'Relatório',
    'fundamentação de direito': 'Fundamentação de Direito',
    'fundamentação de facto': 'Fundamentação de Facto',
    'decisão': 'Decisão',
    'colectivo': 'Coletivo',
    'declaração': 'Declaração',
    'título': "Título"
}

def get_sections(model, judgment_text):
    model.eval()
    output = {"wrapper": "plaintext", "text": judgment_text, "denotations": []}

    ids = list(range(0, len(judgment_text)))


    sections_doc = {"cabeçalho": [], "relatório": [], "delimitação": [], "fundamentação de facto": [],
                    "fundamentação de direito": [], "decisão": [], "colectivo": [], "declaração": [], "foot-note": [],
                    "título": []}
    sections = model.get_sections(judgment_text, "cpu")
    sections = sections[0][1:-1]


    sections_names = []
    for tag in sections:
        sections_names.append(id2word(tag))

    for i, section in enumerate(sections_names):
        if section in ["B-cabeçalho", "I-cabeçalho"]:
            sections_doc["cabeçalho"].append((section, ids[i]))
        elif section in ["B-relatório", "I-relatório"]:
            sections_doc["relatório"].append((section, ids[i]))
        elif section in ["B-delimitação", "I-delimitação"]:
            sections_doc["delimitação"].append((section, ids[i]))
        elif section in ["B-fundamentação-facto", "I-fundamentação-facto"]:
            sections_doc["fundamentação de facto"].append((section, ids[i]))
        elif section in ["B-fundamentação-direito", "I-fundamentação-direito"]:
            sections_doc["fundamentação de direito"].append((section, ids[i]))
        elif section in ["B-decisão", "I-decisão"]:
            sections_doc["decisão"].append((section, ids[i]))
        elif section in ["B-colectivo", "I-colectivo"]:
            sections_doc["colectivo"].append((section, ids[i]))
        elif section in ["B-declaração", "I-declaração"]:
            sections_doc["declaração"].append((section, ids[i]))
        elif section in ["B-foot-note", "I-foot-note"]:
            sections_doc["foot-note"].append((section, ids[i]))
        elif section == "título":
            sections_doc["título"].append((section, ids[i]))


    id = 0
    for key, value in sections_doc.items():

        if len(value) != 0:
            if key in ["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito",
                       "decisão", "foot-note"]:
                output["denotations"].append(
                    {"id": id, "start": value[0][1], "end": value[-1][1], "type": key})
            else:
                zones = []
                for v in value:
                    zones.append(v[1])

                output["denotations"].append({"id": id, "zones": zones, "type": key})
            id += 1

    output["text"] = judgment_text[1:-1]

    return output

def format_text(data):
    text = data['text']
    denotations = data['denotations']
    formatted_text = ""

    for denotation in denotations:
        section_type = section_types[denotation['type']]
        start = denotation.get('start')
        end = denotation.get('end')
        zones = denotation.get('zones')

        if zones:
            for zone in zones:
                formatted_text += f"\n\n{section_type}:\n{text[zone]}"
        else:
            if start is not None and end is not None:
                section_text = "\n".join(text[start:end + 1])
                formatted_text += f"\n\n{section_type}:\n{section_text}"

    return formatted_text

  
def get_important_sections(output, denotations_wanted):


  all_text = ''
  denotations = output["denotations"]

  text = output["text"]

  for d in denotations:
    if d["type"] in denotations_wanted:
      start = d["start"]
      end = d["end"]

      for i,t in enumerate(text):
        if start <= i <= end:
          all_text += t + '\n'

  return all_text 



