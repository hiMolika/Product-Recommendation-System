import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from typing import List, Dict, Tuple, Any
import pickle
import os

class RecommendationEngine:
    def __init__(self, processed_data: pd.DataFrame):
        """Initialize recommendation engine with processed product data"""
        self.data = processed_data
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.product_embeddings = None
        self.tfidf_matrix = None
        self.faiss_index = None
        
    def create_embeddings(self):
        """Create embeddings for all products"""
        print("Creating product embeddings...")
        
        # Create text embeddings using sentence transformer
        search_texts = self.data['search_text'].fillna('').tolist()
        self.product_embeddings = self.sentence_model.encode(search_texts)
        
        # Create TF-IDF matrix for keyword matching
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_texts)
        
        # Create FAISS index for fast similarity search
        dimension = self.product_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.product_embeddings / np.linalg.norm(
            self.product_embeddings, axis=1, keepdims=True
        )
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
        
        print(f"Created embeddings for {len(self.data)} products")
    
    def save_model(self, model_path: str = 'recommendation_model.pkl'):
        """Save trained model components"""
        model_data = {
            'product_embeddings': self.product_embeddings,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'data': self.data
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save FAISS index separately
        faiss.write_index(self.faiss_index, 'faiss_index.index')
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'recommendation_model.pkl'):
        """Load pre-trained model components"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.product_embeddings = model_data['product_embeddings']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.data = model_data['data']
            
            # Load FAISS index
            if os.path.exists('faiss_index.index'):
                self.faiss_index = faiss.read_index('faiss_index.index')
            
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def search_by_query(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search products by text query using semantic similarity"""
        if self.product_embeddings is None:
            print("Embeddings not created. Please run create_embeddings() first.")
            return []
        
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search using FAISS
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        return results
    
    def filter_by_price(self, min_price: float = 0, max_price: float = float('inf')) -> List[int]:
        """Filter products by price range"""
        mask = (self.data['price_numeric'] >= min_price) & (self.data['price_numeric'] <= max_price)
        return self.data[mask].index.tolist()
    
    def filter_by_features(self, features: Dict[str, Any]) -> List[int]:
        """Filter products by specific features"""
        mask = pd.Series([True] * len(self.data))
        
        for feature, value in features.items():
            if feature in self.data.columns:
                if isinstance(value, bool):
                    mask &= (self.data[feature] == value)
                elif isinstance(value, list):
                    # For color filtering
                    if feature == 'colors':
                        color_mask = pd.Series([False] * len(self.data))
                        for color in value:
                            color_mask |= self.data['colors_list'].apply(
                                lambda x: any(color.lower() in str(c).lower() for c in x) if x else False
                            )
                        mask &= color_mask
        
        return self.data[mask].index.tolist()
    
    def get_recommendations(
        self, 
        query: str = "", 
        price_range: Tuple[float, float] = (0, float('inf')),
        features: Dict[str, Any] = None,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """Get product recommendations based on query and filters"""
        
        if features is None:
            features = {}
        
        # Start with all products
        candidate_indices = set(range(len(self.data)))
        
        # Apply price filter
        if price_range != (0, float('inf')):
            price_filtered = set(self.filter_by_price(price_range[0], price_range[1]))
            candidate_indices &= price_filtered
        
        # Apply feature filters
        if features:
            feature_filtered = set(self.filter_by_features(features))
            candidate_indices &= feature_filtered
        
        # If we have a text query, use semantic search
        if query.strip():
            search_results = self.search_by_query(query, top_k * 3)  # Get more results to filter
            
            # Filter search results by other criteria
            filtered_results = []
            for idx, score in search_results:
                if idx in candidate_indices:
                    filtered_results.append((idx, score))
                    if len(filtered_results) >= top_k:
                        break
            
            final_indices = [idx for idx, _ in filtered_results]
        else:
            # If no query, just return top products from filtered set
            final_indices = list(candidate_indices)[:top_k]
        
        # Convert to product dictionaries
        recommendations = []
        for idx in final_indices:
            product = self.data.iloc[idx]
            recommendations.append({
                'id': idx,
                'name': product['name'],
                'price': product['price'],
                'price_numeric': product['price_numeric'],
                'colors': product.get('colors_list', []),
                'plain_text_description': self._truncate_description(product['plain_text_description']),
                'url': product['url'],
                'features': {
                    'portable': product.get('portable', False),
                    'led': product.get('led', False),
                    'height_inches': product.get('height_inches', 0),
                    'acrylic': product.get('acrylic', False),
                    'glass': product.get('glass', False)
                }
            })
        
        return recommendations
    
    def _truncate_description(self, description: str, max_length: int = 150) -> str:
        """Truncate description to specified length"""
        if pd.isna(description):
            return ""
        
        description = str(description).strip()
        if len(description) <= max_length:
            return description
        
        # Find last complete sentence within limit
        truncated = description[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:  # If we can get at least 70% with complete sentence
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def get_similar_products(self, product_id: int, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get products similar to a given product"""
        if self.product_embeddings is None:
            return []
        
        # Get embedding for the target product
        target_embedding = self.product_embeddings[product_id].reshape(1, -1)
        target_embedding = target_embedding / np.linalg.norm(target_embedding, axis=1, keepdims=True)
        
        # Find similar products
        scores, indices = self.faiss_index.search(target_embedding.astype(np.float32), top_k + 1)
        
        # Exclude the target product itself
        similar_products = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != product_id:
                product = self.data.iloc[idx]
                similar_products.append({
                    'id': int(idx),
                    'name': product['name'],
                    'price': product['price'],
                    'similarity_score': float(score),
                    'url': product['url']
                })
                
                if len(similar_products) >= top_k:
                    break
        
        return similar_products

# Example usage and testing
if __name__ == "__main__":
    from data_processor import DataProcessor
    
    # Load and process data
    processor = DataProcessor('products_200.csv')
    processed_data = processor.process_data()
    
    # Create recommendation engine
    engine = RecommendationEngine(processed_data)
    engine.create_embeddings()
    
    # Test recommendations
    print("\nTesting recommendations:")
    
    # Test query-based search
    recommendations = engine.get_recommendations(
        query="portable hookah with LED light",
        top_k=2
    )
    
    for i, rec in enumerate(recommendations):
        print(f"\n{i+1}. {rec['name']}")
        print(f"   Price: {rec['price']}")
        print(f"   plain_text_description: {rec['plain_text_description']}")
        print(f"   URL: {rec['url']}")
    
    # Save the trained model
    engine.save_model()