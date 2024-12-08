"""Version python 3.12"""

# Importing libraries
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------


# Étape 1 : Fonction pour charger un fichier JSONL
def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


# Extraire les descriptions de produits
def extract_descriptions(data):
    descriptions = []
    for item in data:
        desc = " ".join(item.get("description", []))
        if desc:
            descriptions.append(desc)
    return descriptions


# Étape 2 : Prétraitement et segmentation des textes
def preprocess_texts(descriptions, chunk_size=512, chunk_overlap=128):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Convertir chaque description en un objet Document attendu par LangChain
    documents = [Document(page_content=desc) for desc in descriptions]
    return text_splitter.split_documents(documents)


# Étape 3 : Générer les embeddings
def generate_embeddings(
    text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32
):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    texts = [chunk.page_content for chunk in text_chunks]

    # Génération des embeddings en lots
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        embeddings.extend(embeddings_model.embed_documents(batch_texts))

    return embeddings, embeddings_model


# Créer un index vectoriel
def create_vector_store(text_chunks, embeddings_model):
    # Accéder au texte dans l'attribut `page_content` des objets `Document`
    texts = [chunk.page_content for chunk in text_chunks]
    # Créer l'index vectoriel FAISS
    vector_store = FAISS.from_texts(texts, embeddings_model)
    return vector_store


# Étape 4 : Création d'une base de données vectorielle
def create_faiss_vector_store(text_chunks, embeddings_model, faiss_index_path):
    # Récupérer les textes à partir des chunks
    texts = [chunk.page_content for chunk in text_chunks]

    # Créer un index vectoriel avec FAISS
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings_model)

    # Sauvegarder l'index FAISS
    vector_store.save_local(faiss_index_path)
    print(f"Base FAISS sauvegardée dans : {faiss_index_path}")
    return vector_store


# Charger une base de données vectorielle FAISS
def load_faiss_vector_store(faiss_index_path, embeddings):
    # Charger l'index FAISS existant avec l'option de désérialisation
    vector_store = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # Ajout pour autoriser la désérialisation
    )
    print(f"Base FAISS chargée depuis : {faiss_index_path}")
    return vector_store


# Étape 5 : Recherche de produits similaires et configuration du retriever
# Configurer un retriever pour rechercher les descriptions pertinentes
def create_retriever(vector_store):
    retriever = vector_store.as_retriever()
    retriever.search_kwargs.update(
        {"k": 5}
    )  # Limiter à 5 documents les plus pertinents
    return retriever


# Fonction pour formuler le prompt structuré
def create_structured_prompt(retrieved_docs, question):
    documents = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    Vous êtes un assistant intelligent conçu pour répondre aux questions des utilisateurs en utilisant exclusivement les informations contenues dans les documents fournis ci-dessous.
    Vous ne devez **pas** faire appel à vos connaissances internes ni générer d'informations supplémentaires ou incorrectes qui ne sont pas présentes dans ces documents.

    Voici les instructions à suivre :
    1. Utilisez **uniquement les informations présentes dans les documents fournis** pour répondre à la question.
    2. Si une réponse ne peut être trouvée dans les documents, répondez **clairement que vous ne savez pas**.
    3. Lorsque vous fournissez une réponse, **citez précisément les passages** ou les documents d'où vous avez extrait l'information.
    4. **Ne générez pas d'informations sensibles, inappropriées ou potentiellement incorrectes**. Limitez-vous aux informations contenues dans les documents.

    Documents fournis :
    {documents}

    Question :
    {question}

    Veuillez répondre uniquement à partir des documents ci-dessus. Si vous ne trouvez pas d'information pertinente, répondez simplement : "Je ne sais pas."
    """
    return prompt


# Fonction pour interroger le modèle OpenAI
def query_openai(prompt, model="gpt-4", openai_key=None):
    OpenAI.api_key = openai_key
    response = OpenAI.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations des documents fournis.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


# Étape 5 : Recherche et réponse aux requêtes
def search_and_respond(query, retriever, openai_key=None):
    # Récupérer les documents pertinents
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "Je ne sais pas."

    # Créer le prompt structuré
    structured_prompt = create_structured_prompt(retrieved_docs, query)

    # Utiliser OpenAI pour générer une réponse
    response = query_openai(structured_prompt, model="gpt-4", openai_key=openai_key)
    return response


def main():
    # ---------------------------------------------------------------------
    # Initialisation des variables

    # Chemins des fichiers d'entrée
    meta_file_path = "./data/meta.jsonl"

    # Créer le dossier de sortie si nécessaire
    output_dir = "./processed_data"
    os.makedirs(output_dir, exist_ok=True)

    dotenv_path = "./.env"
    faiss_index_path = "./faiss_index"

    # ---------------------------------------------------------------------
    # Traitement des données

    # Charger la clé d'OpenAI
    load_dotenv(dotenv_path, override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("La clé API OpenAI n'est pas définie.")

    # Charger les données
    data = load_data(meta_file_path)
    descriptions = extract_descriptions(data)
    print(f"Nombre de descriptions de produits : {len(descriptions)}")

    # Prétraitement et segmentation des textes
    text_chunks = preprocess_texts(descriptions)
    print(f"Nombre de chunks de texte : {len(text_chunks)}")

    # Générer les embeddings et créer un index vectoriel
    embeddings, embeddings_model = generate_embeddings(text_chunks)
    vector_store = create_vector_store(text_chunks, embeddings_model)
    print("Index vectoriel créé avec succès.")

    # Créer la base de données vectorielle FAISS
    create_faiss_vector_store(text_chunks, embeddings_model, faiss_index_path)

    # Charger l'index FAISS et configurer un retriever
    # vector_store = load_faiss_vector_store(faiss_index_path, embeddings_model)
    retriever = create_retriever(vector_store)

    # Exemple de recherche
    query = "Description du produit OnePlus 6T"
    response = search_and_respond(query, retriever, openai_key)
    print(f"Réponse à la requête '{query}' :\n{response}")


if __name__ == "__main__":
    main()
