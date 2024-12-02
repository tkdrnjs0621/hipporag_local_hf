python ner_passage.py # INPUT : corpus data  OUTPUT : passage ner
python openie_passage.py # INPUT corpus data, passage ner OUTPUT : passage openie
python ner_query.py # INPUT : query data OUTPUT query ner
python build_graph.py # INPUT passage ner passage openie #OUTPUT occurence map (doc, entity -> int) graph (entity-> int)

python encode_and_find_relevant_entities.py # INPUT entity map
python synonym_graph.py

python ircot.py #INPUT graph, entity table, occurrence map #OUTPUT retrieval