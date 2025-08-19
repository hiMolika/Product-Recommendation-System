import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from chatbot_agent import ChatbotAgent
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Hookah Recommendation Chatbot",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f4e79;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .sidebar-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SessionManager:
    """Manage user sessions and conversation persistence"""
    
    @staticmethod
    def get_session_id():
        """Generate or retrieve session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return st.session_state.session_id
    
    @staticmethod
    def save_conversation(session_id: str, conversation_data: dict):
        """Save conversation to file"""
        os.makedirs('conversations', exist_ok=True)
        filename = f'conversations/session_{session_id}.json'
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2, default=str)
    
    @staticmethod
    def load_conversation(session_id: str):
        """Load conversation from file"""
        filename = f'conversations/session_{session_id}.json'
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

@st.cache_data
def load_and_process_data():
    """Load and process product data (cached for performance)"""
    processor = DataProcessor('products_200.csv')
    processed_data = processor.process_data()
    return processed_data

@st.cache_resource
def initialize_recommendation_engine():
    """Initialize recommendation engine (cached for performance)"""
    processed_data = load_and_process_data()
    engine = RecommendationEngine(processed_data)
    
    # Try to load existing model, otherwise create new one
    if not engine.load_model():
        st.info("Training recommendation model... This may take a few moments.")
        engine.create_embeddings()
        engine.save_model()
        st.success("Model trained and saved successfully!")
    
    return engine

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chatbot' not in st.session_state:
        engine = initialize_recommendation_engine()
        st.session_state.chatbot = ChatbotAgent(engine)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False

def display_chat_history():
    """Display the conversation history"""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation")
        
        for i, message in enumerate(st.session_state.conversation_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)

def display_sidebar_info():
    """Display information in the sidebar"""
    st.sidebar.markdown("### 📊 Session Info")
    
    session_id = SessionManager.get_session_id()
    st.sidebar.text(f"Session: {session_id}")
    
    if hasattr(st.session_state, 'chatbot'):
        summary = st.session_state.chatbot.get_conversation_summary()
        
        st.sidebar.markdown("### 🎯 Current Preferences")
        
        prefs = summary['preferences']
        
        if prefs['budget_range'] != "Not specified":
            st.sidebar.text(f"💰 Budget: {prefs['budget_range']}")
        
        if prefs['colors']:
            st.sidebar.text(f"🎨 Colors: {', '.join(prefs['colors'])}")
        
        if prefs['features']:
            st.sidebar.text(f"✨ Features: {', '.join(prefs['features'])}")
        
        if prefs['usage']:
            st.sidebar.text(f"📍 Usage: {prefs['usage']}")
        
        st.sidebar.markdown("### 📈 Stats")
        st.sidebar.metric("Conversation State", summary['state'].title())
        st.sidebar.metric("Messages Exchanged", summary['conversation_length'])
        st.sidebar.metric("Products Shown", summary['recommendations_shown'])

def display_product_analytics():
    """Display product analytics in sidebar"""
    processed_data = load_and_process_data()
    
    st.sidebar.markdown("### 📊 Product Analytics")
    
    # Price distribution
    fig_price = px.histogram(
        processed_data, 
        x='price_category', 
        title='Products by Price Range',
        color='price_category'
    )
    fig_price.update_layout(height=300, showlegend=False)
    st.sidebar.plotly_chart(fig_price, use_container_width=True)
    
    # Feature distribution
    feature_counts = {
        'Portable': processed_data['portable'].sum() if 'portable' in processed_data.columns else 0,
        'LED': processed_data['led'].sum() if 'led' in processed_data.columns else 0,
        'Glass': processed_data['glass'].sum() if 'glass' in processed_data.columns else 0,
        'Acrylic': processed_data['acrylic'].sum() if 'acrylic' in processed_data.columns else 0
    }
    
    fig_features = px.bar(
        x=list(feature_counts.keys()),
        y=list(feature_counts.values()),
        title='Popular Features'
    )
    fig_features.update_layout(height=300)
    st.sidebar.plotly_chart(fig_features, use_container_width=True)

def process_user_message(user_message: str):
    """Process user message and generate response"""
    if user_message.strip():
        # Add user message to history
        st.session_state.conversation_history.append({
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        # Generate bot response
        with st.spinner("🤔 Thinking..."):
            bot_response = st.session_state.chatbot.generate_response(user_message)
        
        # Add bot response to history
        st.session_state.conversation_history.append({
            'type': 'bot',
            'content': bot_response,
            'timestamp': datetime.now()
        })
        
        # Save conversation
        session_id = SessionManager.get_session_id()
        conversation_data = {
            'session_id': session_id,
            'conversation_history': st.session_state.conversation_history,
            'user_preferences': st.session_state.chatbot.get_conversation_summary(),
            'last_updated': datetime.now()
        }
        SessionManager.save_conversation(session_id, conversation_data)

def display_quick_actions():
    """Display quick action buttons"""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Budget Options", use_container_width=True):
            process_user_message("Show me hookahs under 2000 rupees")
            st.rerun()
    
    with col2:
        if st.button("🎒 Portable Hookahs", use_container_width=True):
            process_user_message("I need a portable hookah for travel")
            st.rerun()
    
    with col3:
        if st.button("💡 LED Hookahs", use_container_width=True):
            process_user_message("Show me hookahs with LED lights")
            st.rerun()
    
    with col4:
        if st.button("🏠 Home Use", use_container_width=True):
            process_user_message("Best hookah for home use")
            st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔥 Hookah Recommendation Chatbot</h1>', unsafe_allow_html=True)
    
    # Welcome message for new users
    if not st.session_state.session_started:
        st.markdown("""
        <div class="sidebar-info">
            <h3>Welcome! 👋</h3>
            <p>I'm your personal hookah recommendation assistant. I can help you find the perfect hookah based on:</p>
            <ul>
                <li>💰 Your budget preferences</li>
                <li>🎨 Color and design choices</li>
                <li>🎒 Portability needs</li>
                <li>✨ Special features (LED, unbreakable, etc.)</li>
                <li>📍 Usage context (home, travel, parties)</li>
            </ul>
            <p><strong>Just start chatting below, and I'll guide you to your perfect hookah!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.session_started = True
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display conversation history
        display_chat_history()
        
        # Chat input using st.form (fixes the session state error)
        st.markdown("### 💬 Chat with Assistant")
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your message here...",
                placeholder="e.g., 'I want a portable hookah under 3000 rupees with LED lights'",
                key="user_message_input"
            )
            
            # Submit button
            submitted = st.form_submit_button("Send Message 📤", type="primary", use_container_width=True)
            
            if submitted and user_input.strip():
                process_user_message(user_input.strip())
                st.rerun()
        
        # Additional controls
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.chatbot.reset_conversation()
                st.rerun()
        
        with col_export:
            if st.button("📥 Export Chat", use_container_width=True):
                if st.session_state.conversation_history:
                    conversation_text = ""
                    for msg in st.session_state.conversation_history:
                        role = "You" if msg['type'] == 'user' else "Assistant"
                        conversation_text += f"{role}: {msg['content']}\n\n"
                    
                    st.download_button(
                        label="📄 Download Chat History",
                        data=conversation_text,
                        file_name=f"hookah_chat_{SessionManager.get_session_id()}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        
        # Quick actions
        display_quick_actions()
    
    with col2:
        # Sidebar information
        display_sidebar_info()
        
        # Product analytics
        display_product_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔥 Hookah Recommendation Chatbot | Built with Streamlit & AI</p>
        <p>💡 Pro tip: Be specific about your needs for better recommendations!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()