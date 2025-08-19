import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import ast

class DataProcessor:
    def __init__(self, csv_path: str):
        """Initialize the data processor with CSV file path"""
        self.csv_path = csv_path
        self.df = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} products from CSV")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_price(self, price_str: str) -> float:
        """Extract numeric price from price string"""
        if pd.isna(price_str):
            return 0.0
        
        # Remove 'Rs.' and any non-numeric characters except decimal point
        price_clean = re.sub(r'[^\d.]', '', str(price_str))
        try:
            return float(price_clean)
        except ValueError:
            return 0.0
    
    def parse_colors(self, color_str: str) -> List[str]:
        """Parse color list from string representation"""
        if pd.isna(color_str) or color_str == 'N/A':
            return []
        
        try:
            # Handle string representation of list
            if color_str.startswith('[') and color_str.endswith(']'):
                return ast.literal_eval(color_str)
            else:
                return [color_str.strip()]
        except:
            return []
    
    def extract_features(self, description: str) -> Dict[str, Any]:
        """Extract key features from product description"""
        if pd.isna(description):
            return {}
        
        features = {}
        
        # Extract height information
        height_match = re.search(r'(\d+)\s*inch', description.lower())
        if height_match:
            features['height_inches'] = int(height_match.group(1))
        
        # Check for key features
        features['portable'] = 'portable' in description.lower()
        features['led'] = 'led' in description.lower()
        features['acrylic'] = 'acrylic' in description.lower()
        features['glass'] = 'glass' in description.lower()
        features['unbreakable'] = 'unbreakable' in description.lower()
        features['diffuser'] = 'diffuser' in description.lower()
        features['x_function'] = 'x function' in description.lower()
        
        return features
    
    def create_search_text(self, row: pd.Series) -> str:
        """Create searchable text combining all relevant product info"""
        text_parts = []
        
        # Add product name
        if pd.notna(row['name']):
            text_parts.append(row['name'])
        
        # Add description
        if pd.notna(row['plain_text_description']):
            text_parts.append(row['plain_text_description'])
        
        # Add colors
        colors = self.parse_colors(row['colors'])
        if colors:
            text_parts.append(' '.join(colors))
        
        # Add brand if available
        if pd.notna(row.get('brand', '')):
            text_parts.append(row['brand'])
        
        return ' '.join(text_parts).lower()
    
    def process_data(self) -> pd.DataFrame:
        """Process and enrich the product data"""
        if self.df is None:
            self.load_data()
        
        # Create processed dataframe
        processed_df = self.df.copy()
        
        # Clean and convert price
        processed_df['price_numeric'] = processed_df['price'].apply(self.clean_price)
        
        # Parse colors
        processed_df['colors_list'] = processed_df['colors'].apply(self.parse_colors)
        
        # Extract features
        feature_data = processed_df['plain_text_description'].apply(self.extract_features)
        features_df = pd.json_normalize(feature_data)
        
        # Combine with main dataframe
        processed_df = pd.concat([processed_df, features_df], axis=1)
        
        # Create searchable text
        processed_df['search_text'] = processed_df.apply(self.create_search_text, axis=1)
        
        # Create price categories
        processed_df['price_category'] = pd.cut(
            processed_df['price_numeric'], 
            bins=[0, 1000, 3000, 10000, float('inf')], 
            labels=['Budget', 'Mid-range', 'Premium', 'Luxury']
        )
        
        self.processed_data = processed_df
        print(f"Processed {len(processed_df)} products successfully")
        
        return processed_df
    
    def get_product_summary(self, product_id: int) -> Dict[str, Any]:
        """Get a summary of a specific product"""
        if self.processed_data is None:
            return {}
        
        product = self.processed_data.iloc[product_id]
        
        return {
            'name': product['name'],
            'price': product['price'],
            'price_numeric': product['price_numeric'],
            'colors': product['colors_list'],
            'description': product['plain_text_description'][:200] + '...' if len(str(product['plain_text_description'])) > 200 else product['plain_text_description'],
            'url': product['url'],
            'features': {
                'portable': product.get('portable', False),
                'led': product.get('led', False),
                'height_inches': product.get('height_inches', 0)
            }
        }
    
    def save_processed_data(self, output_path: str = 'processed_products.csv'):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("No processed data to save")

# Example usage and testing
if __name__ == "__main__":
    processor = DataProcessor('products_200.csv')
    df = processor.load_data()
    processed_df = processor.process_data()
    
    # Display sample processed data
    print("\nSample processed data:")
    print(processed_df[['name', 'price_numeric', 'price_category', 'portable', 'led']].head())
    
    processor.save_processed_data()