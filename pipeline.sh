python ner_passage.py
python ner_query.py 

#IF HIPPORAG
    python openie_passage.py
    python build_graph.py
#ELIF ENTITYRAG
    python process_entity_docs.py

python encode_and_find_relevant_entities.py

#IF HIPPORAG 
    python synonym_graph.py
    python hipporag.py
#ELIF ENTITYRAG
    python entityrag.py