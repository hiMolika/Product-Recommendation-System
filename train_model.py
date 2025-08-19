#!/usr/bin/env python3
"""
Training script for the hookah recommendation chatbot
This script processes the data and trains the recommendation model
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from chatbot_agent import ChatbotAgent

def generate_synthetic_training_data(processed_data: pd.DataFrame, num_samples: int = 200) -> pd.DataFrame:
    """Generate synthetic training conversations for the chatbot"""
    
    # Conversation templates
    templates = [
        # Budget-focused queries
        {
            "user_query": "I want a hookah under {price} rupees",
            "intent": "budget_search",
            "price_max": [1000, 1500, 2000, 2500, 3000, 5000]
        },
        {
            "user_query": "Show me budget friendly hookahs",
            "intent": "budget_search",
            "price_max": [2000]
        },
        {
            "user_query": "What's the cheapest hookah you have?",
            "intent": "budget_search",
            "price_max": [1000]
        },
        
        # Feature-focused queries
        {
            "user_query": "I need a portable hookah for travel",
            "intent": "feature_search",
            "features": ["portable"]
        },
        {
            "user_query": "Show me hookahs with LED lights",
            "intent": "feature_search",
            "features": ["led"]
        },
        {
            "user_query": "I want an unbreakable hookah",
            "intent": "feature_search",
            "features": ["unbreakable"]
        },
        {
            "user_query": "Do you have any acrylic hookahs?",
            "intent": "feature_search",
            "features": ["acrylic"]
        },
        
        # Color-focused queries
        {
            "user_query": "I want a {color} hookah",
            "intent": "color_search",
            "colors": ["black", "blue", "red", "green", "purple", "golden", "silver"]
        },
        
        # Combination queries
        {
            "user_query": "Portable {color} hookah under {price} rupees",
            "intent": "combination_search",
            "features": ["portable"],
            "colors": ["black", "blue", "red"],
            "price_max": [2000, 3000, 4000]
        },
        {
            "user_query": "LED hookah for home use around {price}",
            "intent": "combination_search",
            "features": ["led"],
            "usage": ["home"],
            "price_max": [2000, 3000, 4000, 5000]
        },
        
        # Usage-based queries
        {
            "user_query": "Best hookah for {usage}",
            "intent": "usage_search",
            "usage": ["home", "travel", "party", "outdoor"]
        },
        {
            "user_query": "I need a hookah for car trips",
            "intent": "usage_search",
            "usage": ["travel"],
            "features": ["portable"]
        },
        
        # Brand/specific queries
        {
            "user_query": "Show me MYA hookahs",
            "intent": "brand_search",
            "brand": ["MYA"]
        },
        {
            "user_query": "What's your best hookah?",
            "intent": "general_recommendation"
        },
        {
            "user_query": "Recommend me a good hookah",
            "intent": "general_recommendation"
        }
    ]
    
    training_data = []
    
    for i in range(num_samples):
        template = np.random.choice(templates)
        
        # Generate user query
        user_query = template["user_query"]
        
        # Create a dictionary to hold all format values
        format_values = {}
        
        # Fill in placeholders based on what's available in the template
        if "{price}" in user_query and "price_max" in template:
            price = np.random.choice(template["price_max"])
            format_values["price"] = price
        
        if "{color}" in user_query and "colors" in template:
            color = np.random.choice(template["colors"])
            format_values["color"] = color
        
        if "{usage}" in user_query and "usage" in template:
            usage = np.random.choice(template["usage"])
            format_values["usage"] = usage
        
        # Apply all formatting at once
        if format_values:
            try:
                user_query = user_query.format(**format_values)
            except KeyError as e:
                print(f"Warning: Missing format key {e} in template: {user_query}")
                # Skip this sample and continue
                continue
        
        # Create training sample
        sample = {
            "conversation_id": f"train_{i:04d}",
            "user_query": user_query,
            "intent": template["intent"],
            "timestamp": datetime.now(),
        }
        
        # Add template parameters
        for key in ["price_max", "features", "colors", "usage", "brand"]:
            if key in template:
                sample[key] = template[key]
        
        training_data.append(sample)
    
    return pd.DataFrame(training_data)

def evaluate_model_performance(engine: RecommendationEngine, test_queries: list) -> dict:
    """Evaluate the recommendation engine performance"""
    
    results = {
        "total_queries": len(test_queries),
        "successful_recommendations": 0,
        "average_response_time": 0,
        "recommendation_diversity": 0
    }
    
    response_times = []
    all_recommendations = []
    
    for query in test_queries:
        start_time = datetime.now()
        
        try:
            recommendations = engine.get_recommendations(query=query, top_k=2)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            response_times.append(response_time)
            
            if recommendations:
                results["successful_recommendations"] += 1
                all_recommendations.extend([rec["name"] for rec in recommendations])
        
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
    
    # Calculate metrics
    if response_times:
        results["average_response_time"] = np.mean(response_times)
    
    if all_recommendations:
        unique_recommendations = len(set(all_recommendations))
        results["recommendation_diversity"] = unique_recommendations / len(all_recommendations)
    
    results["success_rate"] = results["successful_recommendations"] / results["total_queries"]
    
    return results

def train_and_evaluate():
    """Main training and evaluation function"""
    
    print("🚀 Starting Hookah Recommendation Model Training")
    print("=" * 50)
    
    # Step 1: Load and process data
    print("📊 Step 1: Loading and processing product data...")
    processor = DataProcessor('products_200.csv')
    
    if not os.path.exists('products_200.csv'):
        print("❌ Error: products_200.csv not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
        return False
    
    processed_data = processor.process_data()
    print(f"✅ Processed {len(processed_data)} products successfully")
    
    # Step 2: Generate synthetic training data
    print("\n🤖 Step 2: Generating synthetic training data...")
    training_data = generate_synthetic_training_data(processed_data, num_samples=200)
    print(f"✅ Generated {len(training_data)} training samples")
    
    # Save training data
    training_data.to_csv('training_conversations.csv', index=False)
    print("💾 Training data saved to 'training_conversations.csv'")
    
    # Step 3: Initialize and train recommendation engine
    print("\n🧠 Step 3: Training recommendation engine...")
    engine = RecommendationEngine(processed_data)
    
    start_time = datetime.now()
    engine.create_embeddings()
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Model training completed in {training_time:.2f} seconds")
    
    # Step 4: Save the trained model
    print("\n💾 Step 4: Saving trained model...")
    engine.save_model('recommendation_model.pkl')
    print("✅ Model saved successfully")
    
    # Step 5: Test the chatbot agent
    print("\n🤖 Step 5: Testing chatbot integration...")
    chatbot = ChatbotAgent(engine)
    
    # Test conversations
    test_conversations = [
        "Hi, I need a hookah recommendation",
        "I want something portable under 2000 rupees",
        "Show me hookahs with LED lights",
        "I need a blue hookah for travel"
    ]
    
    print("Testing sample conversations:")
    for query in test_conversations:
        response = chatbot.generate_response(query)
        print(f"User: {query}")
        print(f"Bot: {response[:100]}...")
        print("-" * 30)
        chatbot.reset_conversation()  # Reset for next test
    
    # Step 6: Evaluate model performance
    print("\n📈 Step 6: Evaluating model performance...")
    
    evaluation_queries = [
        "portable hookah",
        "LED light hookah",
        "budget hookah under 1500",
        "blue acrylic hookah",
        "travel hookah",
        "home use hookah",
        "unbreakable hookah",
        "premium hookah"
    ]
    
    performance_metrics = evaluate_model_performance(engine, evaluation_queries)
    
    print("📊 Performance Metrics:")
    print(f"   Success Rate: {performance_metrics['success_rate']*100:.1f}%")
    print(f"   Average Response Time: {performance_metrics['average_response_time']:.3f} seconds")
    print(f"   Recommendation Diversity: {performance_metrics['recommendation_diversity']:.3f}")
    
    # Step 7: Generate model summary
    print("\n📋 Step 7: Model Summary")
    print("=" * 30)
    
    model_info = {
        "products_processed": len(processed_data),
        "training_samples": len(training_data),
        "embedding_dimension": engine.product_embeddings.shape[1] if engine.product_embeddings is not None else 0,
        "training_time_seconds": training_time,
        "model_file_size_mb": os.path.getsize('recommendation_model.pkl') / (1024*1024) if os.path.exists('recommendation_model.pkl') else 0,
        "performance_metrics": performance_metrics
    }
    
    # Save model info
    with open('model_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"✅ Products in database: {model_info['products_processed']}")
    print(f"✅ Training samples generated: {model_info['training_samples']}")
    print(f"✅ Embedding dimensions: {model_info['embedding_dimension']}")
    print(f"✅ Model file size: {model_info['model_file_size_mb']:.2f} MB")
    
    print("\n🎉 Training completed successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")
    
    return True

def quick_test():
    """Quick test function to verify everything works"""
    print("🧪 Running quick functionality test...")
    
    try:
        # Test data loading
        processor = DataProcessor('products_200.csv')
        data = processor.load_data()
        if data is None:
            raise Exception("Failed to load data")
        
        processed_data = processor.process_data()
        print(f"✅ Data processing: {len(processed_data)} products")
        
        # Test recommendation engine
        engine = RecommendationEngine(processed_data)
        
        # Try to load existing model
        if not engine.load_model():
            print("⏳ No existing model found, creating embeddings...")
            engine.create_embeddings()
            engine.save_model()
        
        # Test search
        results = engine.get_recommendations("portable hookah", top_k=2)
        print(f"✅ Search functionality: {len(results)} results returned")
        
        # Test chatbot
        chatbot = ChatbotAgent(engine)
        response = chatbot.generate_response("Hello")
        print(f"✅ Chatbot functionality: Response generated ({len(response)} chars)")
        
        print("🎉 All tests passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hookah recommendation chatbot")
    parser.add_argument("--quick-test", action="store_true", help="Run quick functionality test")
    parser.add_argument("--full-train", action="store_true", help="Run full training process")
    
    args = parser.parse_args()
    
    if args.quick_test:
        success = quick_test()
    elif args.full_train:
        success = train_and_evaluate()
    else:
        print("Please specify either --quick-test or --full-train")
        print("Usage:")
        print("  python train_model.py --quick-test     # Quick functionality test")
        print("  python train_model.py --full-train     # Full training process")
        success = False
    
    sys.exit(0 if success else 1)