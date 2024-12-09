import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F

import open_clip  
from open_clip import create_model_and_transforms, tokenizer
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, send_from_directory

# Initialize Flask app and set up static folder
app = Flask(__name__, static_folder='static')

# Define base directory path for the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(BASE_DIR, 'static')

if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

# Device setup: CUDA if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device).eval()  # Set model to evaluation mode
with open('image_embeddings.pickle', 'rb') as f:
    image_embeddings = pickle.load(f)
tokenizer = open_clip.get_tokenizer(model_name)

def normalize_embedding(embedding):
    """
    Normalize the input embedding to unit length.
    """
    if isinstance(embedding, torch.Tensor):
        return F.normalize(embedding, p=2, dim=-1).cpu().numpy()
    elif isinstance(embedding, np.ndarray):
        return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    else:
        raise TypeError("Embedding must be a tensor or numpy array.")

def compute_pca_embeddings(embeddings, k=30):
    """
    PCA on image embeddings.
    """
    pca = PCA(n_components=k)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca, pca_embeddings

# Precompute PCA transformation for the embeddings
original_embeddings = np.array(image_embeddings['embedding'].tolist())
pca_transformer, pca_embeddings = compute_pca_embeddings(original_embeddings, k=30)

def text_to_image_search(query_embedding, top_k=5, embedding_type='clip', pca_k=30):
    """
    Perform the search from text or image queries to retrieve most similar images.
    """
    try:
        query_embedding = normalize_embedding(query_embedding)

        # Select the appropriate embedding (CLIP or PCA)
        if embedding_type == 'pca':
            embeddings = pca_embeddings
        else:
            embeddings = np.array(image_embeddings['embedding'].tolist())

        # Normalize embeddings and calculate cosine similarity
        normalized_embeddings = np.array([normalize_embedding(emb) for emb in embeddings])

        # If PCA embeddings are used, apply PCA transformation to the query embedding
        if embedding_type == 'pca':
            query_embedding = pca_transformer.transform(query_embedding.reshape(1, -1))

        similarities = cosine_similarity(query_embedding.reshape(1, -1), normalized_embeddings)[0]

        # Get top K similar images
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(image_embeddings['file_name'][i], similarities[i]) for i in top_indices]

    except Exception as e:
        print(f"Error in text_to_image_search: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    Handle search requests based on text and/or image input.
    
    - Extract and process queries (text/image)
    - Perform search and return the top K results.
    """
    # Extract form data
    text_query = request.form.get('text_query')
    image_file = request.files.get('image_query')
    lam = float(request.form.get('lambda', 0.5))
    embedding_choice = request.form.get('embedding_choice', 'clip')
    pca_k = int(request.form.get('pca_k', 50))

    try:
        # Process text query if provided
        text_embedding = None
        if text_query:
            text_tokens = tokenizer([text_query]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens).cpu().detach().numpy()
                text_embedding = normalize_embedding(text_embedding)

        # Process image query if provided
        image_embedding = None
        if image_file:
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image_tensor).cpu().detach().numpy()
                image_embedding = normalize_embedding(image_embedding)

        # Combine text and image embeddings if both are provided
        query_embedding = None
        if text_query and image_file:
            query_embedding = lam * text_embedding + (1.0 - lam) * image_embedding
        elif text_query:
            query_embedding = text_embedding
        elif image_file:
            query_embedding = image_embedding

        if query_embedding is None:
            return jsonify({"error": "Invalid query"}), 400

        # Perform the search and retrieve the top K results
        results = text_to_image_search(query_embedding, top_k=5, embedding_type=embedding_choice, pca_k=pca_k)

        # Return results as JSON with image paths
        results_with_paths = [
            {"image_path": f"static/coco_images_resized/{filename}", "similarity": float(similarity)}
            for filename, similarity in results
        ]
        return jsonify({"results": results_with_paths})

    except Exception as e:
        print(f"Error in search route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
