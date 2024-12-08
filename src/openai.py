"""Version python 3.12"""

# Importing libraries
import json
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.templates import PromptTemplate
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
    text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


# Créer un index vectoriel
def create_vector_store(text_chunks, embeddings):
    # Accéder au texte dans l'attribut `page_content` des objets `Document`
    texts = [chunk.page_content for chunk in text_chunks]
    # Créer l'index vectoriel FAISS
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


# Étape 4 : Création d'une base de données vectorielle
def create_faiss_vector_store(
    text_chunks, embeddings, faiss_index_path
):
    # Récupérer les textes à partir des chunks
    texts = [chunk.page_content for chunk in text_chunks]

    # Créer un index vectoriel avec FAISS
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)

    # Sauvegarder l'index FAISS
    vector_store.save_local(faiss_index_path)
    print(f"Base FAISS sauvegardée dans : {faiss_index_path}")

    return vector_store


# Charger une base de données vectorielle FAISS
def load_faiss_vector_store(faiss_index_path, embeddings):
    # Charger l'index FAISS existant
    vector_store = FAISS.load_local(faiss_index_path, embeddings)
    print(f"Base FAISS chargée depuis : {faiss_index_path}")
    return vector_store


# Étape 5 : Recherche de produits similaires et configuration du retriever
def setup_retriever(faiss_vector_store):
    return faiss_vector_store.as_retriever()


# Fonction pour créer un système de récupération
def create_retrieval_system(faiss_vector_store, openai_api_key):
    # Configurer le retriever
    retriever = setup_retriever(faiss_vector_store)

    # Initialiser un modèle LLM
    llm = OpenAI(api_key=openai_api_key)

    # Définir un prompt pour guider le modèle
    prompt_template = PromptTemplate(
        template=(
            "Voici des descriptions de produits pertinents : {context}\n\n"
            "Question : {question}\n"
            "Réponse :"
        ),
        input_variables=["context", "question"],
    )

    # Créer une chaîne RetrievalQA
    retrieval_system = RetrievalQA(
        llm=llm, retriever=retriever, prompt_template=prompt_template
    )
    return retrieval_system


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

# Récupérer la clé d'OpenAI depuis le fichier .env
load_dotenv(dotenv_path, override=True)
openai_key = os.getenv("OPENAI_API_KEY")

# Charger les données
data = load_data(meta_file_path)
descriptions = extract_descriptions(data)
print(f"Nombre de descriptions de produits : {len(descriptions)}")

# Prétraitement et segmentation des textes
text_chunks = preprocess_texts(descriptions)
print(f"Nombre de chunks de texte : {len(text_chunks)}")

# Générer les embeddings
embeddings = generate_embeddings(text_chunks)
print(f"Modèle utilisé : {embeddings.model_name}")

# Créer un index vectoriel
vector_store = create_vector_store(text_chunks, embeddings)
print(f"Index vectoriel créé : {vector_store}")

faiss_vector_store = create_faiss_vector_store(
    text_chunks, embeddings, faiss_index_path
)
print("Base de données vectorielle FAISS créée avec succès.")

# faiss_vector_store = load_faiss_vector_store(faiss_index_path, embeddings)
# print("Base de données vectorielle FAISS chargée avec succès.")

# Créez le système de récupération
retrieval_system = create_retrieval_system(faiss_vector_store, openai_key)
print("Système de récupération créé avec succès.")
