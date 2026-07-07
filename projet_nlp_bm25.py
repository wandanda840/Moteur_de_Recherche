import spacy
from rank_bm25 import BM25Okapi

QRY_PATH="CISI/DATA/CISI_dev.QRY"
EXPORT_REL_PATH="CISI_results_bm25.REL"



nlp = spacy.load("en_core_web_sm")

def parse_documents(filepath):
    documents = {}
    current_id = None #id du document en cours de lecture
    current_text = [] #la ligne du document en cours de lecture

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() #on enlève les espaces au début et à la fin de la ligne
            if line.startswith('.I'): #si la ligne commence par .I le document commence
                if current_id is not None:
                    documents[current_id] = ' '.join(current_text) #on stocke le document précédent avant de passer au suivant
                current_id = int(line.split()[1])
                current_text = [] #on réinitialise le texte courant pour le nouveau document
            
            elif line.startswith('.'): #Bon ça c'est pas pour les textes mais je vois que dans les queries il y a des lignes qui commencent par .T et autres
                pass
            else:
                if line:
                    current_text.append(line)

        if current_id is not None:
            documents[current_id] = ' '.join(current_text)

    return documents

def tokenize(text):
    doc = nlp(text) #on parse le texte avec spacy pour obtenir une liste de tokens et leurs propriétés (stop words, ponctuation, lemme,...)
    tokens = []
    for token in doc:
        if (not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) > 1):
            tokens.append(token.lemma_.lower())
    return tokens



########## CODE PRINCIPAL ############

# on parse
docs = parse_documents("CISI/DATA/CISI.ALLnettoye")

#docs contient un dictionnaire {id_doc : texte_du_doc} pour tous les documents du dataset

print(f"Nombre de docs : {len(docs)}")
print(f"Les 200 premiers caractères du document 1 : {docs[1][:200]}")

#On tokenise maintenant tous les document de docs avec la fonction tokenize qu'on a écrit plus haut 
tokenized_docs = {}
for doc_id, text in docs.items():
    tokenized_docs[doc_id] = tokenize(text)

# vérification
print(f"Nombre de docs tokenisés : {len(tokenized_docs)}") #On devrait avoir le même nombre de documents que dans docs
print(f"Doc 1 les 20 premiers tokens : {tokenized_docs[1][:20]}")
print(f"Doc 2 les 20 premiers tokens : {tokenized_docs[2][:20]}")


from sklearn.feature_extraction.text import TfidfVectorizer

# on recolle les listes de tokens en string pour scikit-learn on transforme la liste de liste de tokens en un liste de string 

corpus = {}
for doc_id, tokens in tokenized_docs.items():
    corpus[doc_id] = ' '.join(tokens)
#corpus contient un dictionnaire {id_doc : texte_tokenisé_du_doc (et non plus jsute un liste de tokens) } pour tous les documents du dataset

# on garde l'ordre des ids pour retrouver quel vecteur correspond à quel doc parceque scikit-learn va nous retourner une matrice où chaque ligne correspond à un document dans l'ordre où on lui a donné les textes
doc_ids = list(corpus.keys())

bm25_corpus = [tokenized_docs[doc_id] for doc_id in doc_ids]
bm25 = BM25Okapi(bm25_corpus)

texts = [corpus[doc_id] for doc_id in doc_ids] #texts contient une liste de tous les textes tokenisés ces textes sont des chaines de caractères
#texts est un liste de string

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

print(f"Nombre de documents : {tfidf_matrix.shape[0]}")
print(f"Taille du vocabulaire : {tfidf_matrix.shape[1]}")



import numpy as np

# le vocabulaire : chaque mot a un indice numérique
vocabulary = vectorizer.get_feature_names_out()

# index inversé : { "library" : {1: 0.42, 5: 0.31, 23: 0.18}, ... }
# pour chaque terme -> les docs qui le contiennent avec leur poids TF-IDF
inverted_index = {}

# on parcourt la matrice colonne par colonne (un terme = une colonne)
cx = tfidf_matrix.tocsc()  # on convertit en format colonne pour parcourir efficacement

for term_idx, term in enumerate(vocabulary):
    col = cx.getcol(term_idx)
    # on récupère les docs où ce terme a un poids > 0
    doc_indices = col.nonzero()[0]
    if len(doc_indices) > 0:
        inverted_index[term] = [
            (doc_ids[i], tfidf_matrix[i, term_idx])
            for i in doc_indices
        ]

#inverted_index contient un dictionnaire {terme : [(id_doc1, poids_tfidf), (id_doc2, poids_tfidf),...]} uniquement pour les docs qui ont le terme


# print(f"Nombre de termes dans l'index : {len(inverted_index)}")
# print(f"Docs contenant 'library' : {(inverted_index['library'][:10])}")


queries = parse_documents(QRY_PATH) #retourne dans un dictionnaire {id_query : texte_query}

#On tockenise aussi les queris comme on l'a fait avec les documents


tokenized_queries = {}


for query_id, text in queries.items():
    tokenized_queries[query_id] = tokenize(text)  
    
#tokenized_queries contient un dictionnaire {id_query : liste_de_tokens_de_la_query} pour toutes les queries du dataset
    
    

print("############# Vérification des queries tokenisées ##################")

print(f"Nombre de queries tokenisées : {len(tokenized_queries)}")
print(f"Query 1 les 20 premiers tokens : {tokenized_queries[1][:20]}")
print(f"Query 2 les 20 premiers tokens : {tokenized_queries[2][:20]}")


from sklearn.feature_extraction.text import TfidfVectorizer

# on recolle les tokens en string pour scikit-learn comme c'est ça qu'il prend ene entree on transforme la liste de liste de tokens en une liste de string 

query_corpus = {}
for query_id, tokens in tokenized_queries.items():
    query_corpus[query_id] = ' '.join(tokens)

# on garde l'ordre des ids pour retrouver quel vecteur correspond à quel doc
query_ids = list(query_corpus.keys())
query_texts = [query_corpus[query_id] for query_id in query_ids]


##### Maintenant on met en place le moteur de rechercher ######

from sklearn.metrics.pairwise import cosine_similarity

#Pour chaque query on calcule la similarité cosinus entre le vecteur de la query et les vecteurs des documents qui contienent les termes de la query (en utilisant l'index inversé pour ne calculer que sur les docs pertinents)

def search(query_id):
    query_tokens = tokenized_queries[query_id]

    relevant_docs = set()
    for term in query_tokens:
        if term in inverted_index:
            relevant_docs.update([doc_id for doc_id, _ in inverted_index[term]])

    if not relevant_docs:
        return []
    
    #relevant_docs contient l'ensemble des ids de documents qui pourraent être pertinents pour la query (basé uniquement sur le nombre d'occurences des termes de la query dans les documents, pas encore de score de pertinence)


    scores = bm25.get_scores(query_tokens)
    #déjà bm25 contient les vecteurs de tous les documents et on lui donne les tokens de la query pour qu'il nous retourne une liste de scores pour tous les documents


    doc_scores = [
        (doc_id, scores[doc_ids.index(doc_id)])
        for doc_id in relevant_docs
    ]

    ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    
    ranked_docs = [(doc_id, score) for doc_id, score in ranked_docs if score > 0]

    return ranked_docs[:50]

#La TF_IDF a juste servi pour le filtrage


def generate_REL_file(rel_path):
    with open(rel_path, 'w', encoding='utf-8') as f:
        for query_id in queries.keys():
            results = search(query_id)
            for doc_id, score in results:
                f.write(f"{query_id} {doc_id} {score:.4f}\n")
                


generate_REL_file(EXPORT_REL_PATH)
print(f"Fichier REL généré : {EXPORT_REL_PATH}")