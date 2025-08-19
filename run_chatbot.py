#!/usr/bin/env python3
"""
Simple launcher script for the Hookah Recommendation Chatbot
This makes it easy for beginners to get started
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 HOOKAH RECOMMENDATION CHATBOT LAUNCHER")
    print("=" * 60)
    print("This script will help you set up and run your chatbot!")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        print("Please install a newer version of Python.")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'data_processor.py',
        'recommendation_engine.py',
        'chatbot_agent.py',
        'train_model.py',
        'requirements.txt',
        'products_200.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please ensure all files are in the same directory.")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out. Please try manual installation.")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def train_model():
    """Train the recommendation model"""
    print("\n🧠 Training the AI model...")
    print("This will process your product data and create the recommendation engine.")
    
    try:
        # First try quick test
        print("Running quick functionality test...")
        result = subprocess.run([
            sys.executable, 'train_model.py', '--quick-test'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print("Quick test failed, running full training...")
            result = subprocess.run([
                sys.executable, 'train_model.py', '--full-train'
            ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out. This might indicate a problem.")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def launch_chatbot():
    """Launch the Streamlit chatbot application"""
    print("\n🚀 Launching the chatbot application...")
    print("Your browser should open automatically.")
    print("If not, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped. Thanks for using the Hookah Recommendation Chatbot!")
    except Exception as e:
        print(f"❌ Failed to launch chatbot: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Step 1: Check Python version
    print("🔍 Step 1: Checking Python version...")
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Step 2: Check required files
    print("\n🔍 Step 2: Checking required files...")
    if not check_files():
        input("Press Enter to exit...")
        return
    
    # Step 3: Ask user what they want to do
    print("\n🎯 What would you like to do?")
    print("1. 🆕 First time setup (install packages + train model + run)")
    print("2. 🧠 Just train/retrain the model")
    print("3. 🚀 Just run the chatbot (if already set up)")
    print("4. 📦 Just install packages")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Full setup
            if install_requirements() and train_model():
                launch_chatbot()
            break
            
        elif choice == '2':
            # Just train model
            if train_model():
                print("\n✅ Model training complete!")
                run_app = input("Would you like to run the chatbot now? (y/n): ").strip().lower()
                if run_app in ['y', 'yes']:
                    launch_chatbot()
            break
            
        elif choice == '3':
            # Just run chatbot
            if os.path.exists('recommendation_model.pkl'):
                launch_chatbot()
            else:
                print("❌ No trained model found. Please run option 1 or 2 first.")
            break
            
        elif choice == '4':
            # Just install packages
            install_requirements()
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def quick_start():
    """Quick start function for experienced users"""
    print_banner()
    print("🚀 QUICK START MODE")
    print("This will automatically set up everything for you!")
    
    steps = [
        ("Checking files", check_files),
        ("Installing packages", install_requirements),
        ("Training model", train_model),
    ]
    
    for step_name, step_func in steps:
        print(f"\n⏳ {step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed. Please run the manual setup.")
            return False
    
    print("\n🎉 Setup complete! Launching chatbot...")
    time.sleep(2)
    launch_chatbot()
    return True

if __name__ == "__main__":
    # Check if user wants quick start
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_start()
    else:
        main()