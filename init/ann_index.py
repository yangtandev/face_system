# init/ann_index.py
import faiss
import numpy as np
import os
import json

class AnnIndex:
    """
    A manager for a Faiss index for Approximate Nearest Neighbor (ANN) search.
    This class handles building, saving, loading, and searching the index.
    """
    def __init__(self, dimension=512, index_path="media/faiss.index", mapping_path="media/faiss_map.json"):
        """
        Initializes the AnnIndex manager.

        Args:
            dimension (int): The dimension of the feature vectors (e.g., 512 for ResNet).
            index_path (str): Path to save/load the Faiss index file.
            mapping_path (str): Path to save/load the index-to-person_id mapping.
        """
        self.dimension = dimension
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index = None
        self.id_map = []  # List where the index is the Faiss ID and the value is the person_id

    def build(self, features_dict):
        """
        Builds the Faiss index from a dictionary of features.

        Args:
            features_dict (dict): A dictionary where keys are person_id and values are lists of embeddings.
        """
        print("Building new Faiss index...")
        # Using IndexFlatIP: for normalized vectors, Inner Product is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []
        
        all_embeddings = []

        for person_id, embeddings in features_dict.items():
            if person_id == "id_name" or not embeddings:
                continue
            
            # Normalize embeddings before adding to the index
            for emb in embeddings:
                normalized_emb = emb / np.linalg.norm(emb)
                all_embeddings.append(normalized_emb)
                self.id_map.append(person_id)

        if not all_embeddings:
            print("No embeddings found to build the index.")
            return

        # Faiss requires a numpy array of float32
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        self.index.add(embeddings_matrix)
        
        print(f"Faiss index built successfully with {self.index.ntotal} vectors.")
        self.save()

    def load(self):
        """
        Loads the index and the ID mapping from disk.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                print(f"Loading existing Faiss index from {self.index_path}...")
                self.index = faiss.read_index(self.index_path)
                
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self.id_map = json.load(f)
                
                print(f"Faiss index loaded successfully with {self.index.ntotal} vectors.")
                return True
            except Exception as e:
                print(f"Error loading Faiss index: {e}. A new index will be built.")
                self.index = None
                self.id_map = []
                return False
        return False

    def save(self):
        """
        Saves the index and the ID mapping to disk.
        """
        if self.index is None:
            print("Cannot save, index is not built.")
            return
            
        print(f"Saving Faiss index to {self.index_path}...")
        # Ensure media directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_map, f)
        print("Faiss index and ID map saved.")

    def search(self, vector, k=5):
        """
        Searches the index for the k nearest neighbors.

        Args:
            vector (np.ndarray): The query vector.
            k (int): The number of nearest neighbors to find.

        Returns:
            tuple: A tuple containing (distances, person_ids) for the nearest neighbors.
                   Returns (None, None) if the index is not ready.
        """
        if self.index is None or self.index.ntotal == 0:
            return None, None
        
        # Normalize the query vector
        query_vector = vector / np.linalg.norm(vector)
        query_vector = np.array([query_vector]).astype('float32')

        distances, indices = self.index.search(query_vector, k)
        
        # Map indices back to person_ids
        person_ids = [self.id_map[i] for i in indices[0]]
        
        return distances[0], person_ids
