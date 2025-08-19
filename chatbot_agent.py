import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class ConversationState(Enum):
    GREETING = "greeting"
    UNDERSTANDING_NEEDS = "understanding_needs"
    SHOWING_PRODUCTS = "showing_products"
    GATHERING_PREFERENCES = "gathering_preferences"
    FINALIZING = "finalizing"

@dataclass
class UserPreferences:
    budget_min: float = 0
    budget_max: float = float('inf')
    preferred_colors: List[str] = None
    size_preference: str = ""  # portable, large, etc.
    features_wanted: List[str] = None
    brand_preference: str = ""
    usage_context: str = ""  # home, travel, party, etc.
    
    def __post_init__(self):
        if self.preferred_colors is None:
            self.preferred_colors = []
        if self.features_wanted is None:
            self.features_wanted = []

class ChatbotAgent:
    def __init__(self, recommendation_engine):
        """Initialize chatbot with recommendation engine"""
        self.recommendation_engine = recommendation_engine
        self.conversation_state = ConversationState.GREETING
        self.user_preferences = UserPreferences()
        self.conversation_history = []
        self.last_recommendations = []
        self.context_memory = {}
        
        # Intent patterns
        self.intent_patterns = {
            'greeting': [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\bwhat\s+can\s+you\s+do\b',
                r'\bhelp\s+me\b'
            ],
            'budget': [
                r'\b(\d+)\s*(rupees?|rs\.?|inr)\b',
                r'\bunder\s+(\d+)\b',
                r'\bbetween\s+(\d+)\s+and\s+(\d+)\b',
                r'\bbudget\s+is\s+(\d+)\b',
                r'\bcheap|budget|affordable|expensive|premium\b'
            ],
            'color': [
                r'\b(black|white|blue|red|green|purple|golden?|silver|pink|brown|rose\s+gold)\b',
                r'\bcolor\s+(preference|choice)\b'
            ],
            'features': [
                r'\bportable\b',
                r'\bled\s+light\b',
                r'\bunbreakable\b',
                r'\bglass\s+base\b',
                r'\bacrylic\b',
                r'\bdiffuser\b',
                r'\bx\s+function\b',
                r'\bcompact\b',
                r'\btall|large|small|mini\b'
            ],
            'usage': [
                r'\b(travel|trip|tour|car|home|party|outdoor|indoor)\b',
                r'\bfor\s+(travel|home|party)\b'
            ],
            'product_inquiry': [
                r'\bshow\s+me\b',
                r'\brecommend\b',
                r'\bsuggest\b',
                r'\bwhat\s+do\s+you\s+have\b',
                r'\bbest\s+hookah\b'
            ],
            'more_info': [
                r'\btell\s+me\s+more\b',
                r'\bdetails?\b',
                r'\bfeatures?\b',
                r'\bspecifications?\b'
            ],
            'affirmative': [
                r'\b(yes|yeah|yep|sure|okay|ok|sounds?\s+good)\b'
            ],
            'negative': [
                r'\b(no|nope|not\s+really|not\s+interested)\b'
            ]
        }
    
    def extract_intent(self, user_input: str) -> Dict[str, Any]:
        """Extract user intent and entities from input"""
        user_input_lower = user_input.lower()
        intents = {}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, user_input_lower)
                if matches:
                    intents[intent] = matches
        
        return intents
    
    def extract_budget(self, user_input: str) -> Tuple[float, float]:
        """Extract budget information from user input"""
        budget_min, budget_max = self.user_preferences.budget_min, self.user_preferences.budget_max
        
        # Look for specific numbers
        numbers = re.findall(r'\d+', user_input)
        
        if 'under' in user_input.lower() and numbers:
            budget_max = float(numbers[0])
        elif 'between' in user_input.lower() and len(numbers) >= 2:
            budget_min = float(numbers[0])
            budget_max = float(numbers[1])
        elif numbers:
            # If just one number mentioned, treat as max budget
            budget_max = float(numbers[0])
        
        # Handle qualitative terms
        if any(term in user_input.lower() for term in ['cheap', 'budget', 'affordable']):
            budget_max = min(budget_max, 2000)
        elif any(term in user_input.lower() for term in ['expensive', 'premium', 'high-end']):
            budget_min = max(budget_min, 3000)
        
        return budget_min, budget_max
    
    def extract_colors(self, user_input: str) -> List[str]:
        """Extract color preferences from user input"""
        colors = []
        color_patterns = [
            r'\b(black|white|blue|red|green|purple|golden?|silver|pink|brown)\b',
            r'\brose\s+gold\b'
        ]
        
        for pattern in color_patterns:
            matches = re.findall(pattern, user_input.lower())
            colors.extend(matches)
        
        return list(set(colors))  # Remove duplicates
    
    def extract_features(self, user_input: str) -> List[str]:
        """Extract feature preferences from user input"""
        features = []
        
        feature_mapping = {
            'portable': ['portable', 'travel', 'compact', 'small'],
            'led': ['led', 'light', 'lighting'],
            'unbreakable': ['unbreakable', 'durable', 'strong'],
            'glass': ['glass'],
            'acrylic': ['acrylic', 'plastic'],
            'diffuser': ['diffuser', 'smooth smoke'],
            'x_function': ['x function', 'no pressure']
        }
        
        user_input_lower = user_input.lower()
        
        for feature, keywords in feature_mapping.items():
            if any(keyword in user_input_lower for keyword in keywords):
                features.append(feature)
        
        return features
    
    def update_preferences(self, user_input: str):
        """Update user preferences based on input"""
        # Update budget
        budget_min, budget_max = self.extract_budget(user_input)
        if budget_min != self.user_preferences.budget_min or budget_max != self.user_preferences.budget_max:
            self.user_preferences.budget_min = budget_min
            self.user_preferences.budget_max = budget_max
        
        # Update colors
        colors = self.extract_colors(user_input)
        if colors:
            self.user_preferences.preferred_colors.extend(colors)
            self.user_preferences.preferred_colors = list(set(self.user_preferences.preferred_colors))
        
        # Update features
        features = self.extract_features(user_input)
        if features:
            self.user_preferences.features_wanted.extend(features)
            self.user_preferences.features_wanted = list(set(self.user_preferences.features_wanted))
        
        # Update usage context
        usage_keywords = ['travel', 'home', 'party', 'outdoor', 'car', 'tour']
        for keyword in usage_keywords:
            if keyword in user_input.lower():
                self.user_preferences.usage_context = keyword
                break
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get product recommendations based on current preferences"""
        # Build query from preferences
        query_parts = []
        
        if self.user_preferences.features_wanted:
            query_parts.extend(self.user_preferences.features_wanted)
        
        if self.user_preferences.usage_context:
            query_parts.append(self.user_preferences.usage_context)
        
        if self.user_preferences.preferred_colors:
            query_parts.extend(self.user_preferences.preferred_colors)
        
        query = ' '.join(query_parts)
        
        # Build features dict
        features = {}
        for feature in self.user_preferences.features_wanted:
            if feature in ['portable', 'led', 'acrylic', 'glass', 'unbreakable', 'diffuser']:
                features[feature] = True
        
        if self.user_preferences.preferred_colors:
            features['colors'] = self.user_preferences.preferred_colors
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(
            query=query,
            price_range=(self.user_preferences.budget_min, self.user_preferences.budget_max),
            features=features,
            top_k=2
        )
        
        self.last_recommendations = recommendations
        return recommendations
    
    def generate_response(self, user_input: str) -> str:
        """Generate chatbot response based on user input and conversation state"""
        # Add to conversation history
        self.conversation_history.append({"user": user_input, "timestamp": pd.Timestamp.now()})
        
        # Extract intent and update preferences
        intents = self.extract_intent(user_input)
        self.update_preferences(user_input)
        
        response = ""
        
        # State machine logic
        if self.conversation_state == ConversationState.GREETING:
            if 'greeting' in intents or any(word in user_input.lower() for word in ['hi', 'hello', 'hey']):
                response = self._handle_greeting()
                self.conversation_state = ConversationState.UNDERSTANDING_NEEDS
            else:
                response = self._handle_greeting()
                self.conversation_state = ConversationState.UNDERSTANDING_NEEDS
                # Also process the input as if it's a need
                response += "\n\n" + self._handle_understanding_needs(user_input)
        
        elif self.conversation_state == ConversationState.UNDERSTANDING_NEEDS:
            response = self._handle_understanding_needs(user_input)
            
            # Check if we have enough info to show products
            if (self.user_preferences.budget_max < float('inf') or 
                self.user_preferences.features_wanted or 
                'product_inquiry' in intents):
                self.conversation_state = ConversationState.SHOWING_PRODUCTS
                response += "\n\n" + self._show_recommendations()
        
        elif self.conversation_state == ConversationState.SHOWING_PRODUCTS:
            if 'more_info' in intents:
                response = self._provide_more_details()
            elif 'negative' in intents:
                response = self._handle_dissatisfaction()
                self.conversation_state = ConversationState.GATHERING_PREFERENCES
            elif 'affirmative' in intents:
                response = self._handle_satisfaction()
                self.conversation_state = ConversationState.FINALIZING
            else:
                response = self._gather_more_preferences()
                self.conversation_state = ConversationState.GATHERING_PREFERENCES
        
        elif self.conversation_state == ConversationState.GATHERING_PREFERENCES:
            response = self._handle_preference_gathering(user_input)
            if self._has_sufficient_preferences():
                self.conversation_state = ConversationState.SHOWING_PRODUCTS
                response += "\n\n" + self._show_recommendations()
        
        elif self.conversation_state == ConversationState.FINALIZING:
            response = self._handle_finalization(user_input)
        
        # Add response to history
        self.conversation_history.append({"bot": response, "timestamp": pd.Timestamp.now()})
        
        return response
    
    def _handle_greeting(self) -> str:
        """Handle greeting messages"""
        return """ Hello! I'm your hookah recommendation assistant. I'm here to help you find the perfect hookah based on your preferences and budget.

I can help you with:
 Finding hookahs based on your needs
 Budget-friendly recommendations
 Color and design preferences
 Portable vs home hookah options
 LED and special features

What kind of hookah are you looking for today?"""
    
    def _handle_understanding_needs(self, user_input: str) -> str:
        """Handle understanding user needs"""
        response_parts = []
        
        # Acknowledge what we understood
        if self.user_preferences.budget_max < float('inf'):
            if self.user_preferences.budget_min > 0:
                response_parts.append(f" I see you're looking for something between ₹{int(self.user_preferences.budget_min)} - ₹{int(self.user_preferences.budget_max)}")
            else:
                response_parts.append(f" I understand your budget is around ₹{int(self.user_preferences.budget_max)}")
        
        if self.user_preferences.preferred_colors:
            colors_str = ", ".join(self.user_preferences.preferred_colors)
            response_parts.append(f" You prefer {colors_str} colors")
        
        if self.user_preferences.features_wanted:
            features_str = ", ".join(self.user_preferences.features_wanted)
            response_parts.append(f" You're interested in: {features_str}")
        
        if self.user_preferences.usage_context:
            response_parts.append(f" For {self.user_preferences.usage_context} use")
        
        if response_parts:
            response = "Great! " + ". ".join(response_parts) + "."
        else:
            response = "I'd love to help you find the perfect hookah!"
        
        # Ask for more info if needed
        if not self._has_sufficient_preferences():
            response += "\n\nTo give you the best recommendations, could you tell me:"
            
            if self.user_preferences.budget_max == float('inf'):
                response += "\n• What's your budget range?"
            
            if not self.user_preferences.features_wanted:
                response += "\n• Do you need it to be portable for travel?"
                response += "\n• Are you interested in LED lighting?"
            
            if not self.user_preferences.usage_context:
                response += "\n• Will you use it at home, for travel, or parties?"
        
        return response
    
    def _show_recommendations(self) -> str:
        """Show product recommendations"""
        recommendations = self.get_recommendations()
        
        if not recommendations:
            return "I couldn't find any products matching your exact criteria. Let me adjust the search parameters and try again."
        
        response = " **Here are my top 2 recommendations for you:**\n\n"
        
        for i, product in enumerate(recommendations, 1):
            response += f"**{i}. {product['name']}**\n"
            response += f" Price: {product['price']}\n"
            
            if product['colors']:
                colors_str = ", ".join([str(c) for c in product['colors']])
                response += f" Available Colors: {colors_str}\n"
            
            # Highlight key features
            features = []
            if product['features'].get('portable'):
                features.append(" Portable")
            if product['features'].get('led'):
                features.append(" LED Light")
            if product['features'].get('unbreakable'):
                features.append(" Unbreakable")
            if product['features'].get('height_inches'):
                features.append(f" {product['features']['height_inches']}\" tall")
            
            if features:
                response += f" Features: {', '.join(features)}\n"
            
            response += f" {product['plain_text_description']}\n"
            response += f" **[View Product]({product['url']})**\n\n"
        
        response += " **What would you like to know more about?**\n"
        response += "• More details about any of these products?\n"
        response += "• Different price range or features?\n"
        response += "• Other color options?"
        
        return response
    
    def _handle_preference_gathering(self, user_input: str) -> str:
        """Handle gathering more specific preferences"""
        response = "Thanks for the additional information! "
        
        # Acknowledge new preferences
        new_prefs = []
        if 'budget' in self.extract_intent(user_input):
            new_prefs.append("budget updated")
        if self.extract_colors(user_input):
            new_prefs.append("color preferences noted")
        if self.extract_features(user_input):
            new_prefs.append("feature requirements understood")
        
        if new_prefs:
            response += f"I've {', '.join(new_prefs)}. "
        
        response += "Let me find better matches for you."
        return response
    
    def _provide_more_details(self) -> str:
        """Provide more details about recommended products"""
        if not self.last_recommendations:
            return "Let me first show you some recommendations based on your preferences."
        
        response = " **Detailed Information:**\n\n"
        
        for i, product in enumerate(self.last_recommendations, 1):
            response += f"**{i}. {product['name']} - {product['price']}**\n"
            
            # Technical details
            if product['features'].get('height_inches'):
                response += f"• Height: {product['features']['height_inches']} inches\n"
            
            # Material information
            materials = []
            if product['features'].get('acrylic'):
                materials.append("Acrylic")
            if product['features'].get('glass'):
                materials.append("Glass")
            if materials:
                response += f"• Material: {', '.join(materials)}\n"
            
            # Special features
            special_features = []
            if product['features'].get('portable'):
                special_features.append("Easy to carry and transport")
            if product['features'].get('led'):
                special_features.append("LED lighting system")
            if product['features'].get('diffuser'):
                special_features.append("Diffuser for smooth smoke")
            
            if special_features:
                response += f"• Special Features: {', '.join(special_features)}\n"
            
            response += f"• Link: {product['url']}\n\n"
        
        response += "Would you like to proceed with any of these, or would you prefer to see different options?"
        
        return response
    
    def _handle_dissatisfaction(self) -> str:
        """Handle when user is not satisfied with recommendations"""
        response = "No worries! Let me help you find something better. "
        
        # Ask specific questions to refine
        questions = []
        
        if self.user_preferences.budget_max == float('inf'):
            questions.append("What's your preferred price range?")
        
        if not self.user_preferences.preferred_colors:
            questions.append("Do you have any color preferences?")
        
        if not self.user_preferences.features_wanted:
            questions.append("What features are most important to you? (portable, LED lights, etc.)")
        
        if questions:
            response += "Could you help me understand:\n• " + "\n• ".join(questions)
        else:
            response += "What specifically would you like to be different? (price, size, features, etc.)"
        
        return response
    
    def _handle_satisfaction(self) -> str:
        """Handle when user is satisfied with recommendations"""
        response = "Excellent! I'm glad I could help you find the right hookah. "
        
        if self.last_recommendations:
            response += "\n\n **Your Selected Options:**\n"
            for i, product in enumerate(self.last_recommendations, 1):
                response += f"{i}. **{product['name']}** - {product['price']}\n"
                response += f"   🔗 [Order Now]({product['url']})\n"
        
        response += "\n💡 **Pro Tips:**\n"
        response += "• Check product reviews before ordering\n"
        response += "• Consider buying flavor and charcoal together\n"
        response += "• Follow proper cleaning instructions for longevity\n"
        
        response += "\nIs there anything else I can help you with today?"
        
        return response
    
    def _gather_more_preferences(self) -> str:
        """Ask for more specific preferences"""
        return """I'd like to understand your preferences better to give you more targeted recommendations.

Could you tell me more about:
 **Primary Use**: Home sessions, travel, parties?
 **Budget Range**: What's comfortable for you to spend?
 **Style Preference**: Traditional glass or modern acrylic?
 **Portability**: Important for you or not necessary?
 **Special Features**: LED lights, unbreakable design, etc.?

Just mention what's important to you, and I'll find the perfect match!"""
    
    def _handle_finalization(self, user_input: str) -> str:
        """Handle final conversation steps"""
        if any(word in user_input.lower() for word in ['thank', 'thanks', 'bye', 'goodbye']):
            return """Thank you for choosing our hookah recommendation service! 

I hope you enjoy your new hookah. Feel free to come back anytime if you need:
• Accessories recommendations
• Different hookah options
• Help with any hookah-related questions

Have a great smoking session! """
        
        elif 'help' in user_input.lower() or '?' in user_input:
            return self._provide_additional_help()
        
        else:
            return "Is there anything else I can help you with regarding hookahs or these recommendations?"
    
    def _provide_additional_help(self) -> str:
        """Provide additional help information"""
        return """ **I can help you with:**

 **Product Search**: Find hookahs by features, price, or brand
 **Budget Recommendations**: Best options in your price range
 **Style Matching**: Colors and designs that suit your taste
 **Usage-based**: Home, travel, or party-specific recommendations
 **Feature Comparison**: LED, portable, unbreakable options

**Just ask me things like:**
• "Show me portable hookahs under ₹2000"
• "I want a blue hookah with LED lights"
• "Best hookah for home use"
• "Unbreakable hookah for travel"

What would you like to explore?"""
    
    def _has_sufficient_preferences(self) -> bool:
        """Check if we have enough preferences to make good recommendations"""
        return (
            self.user_preferences.budget_max < float('inf') or
            len(self.user_preferences.features_wanted) >= 1 or
            len(self.user_preferences.preferred_colors) >= 1 or
            self.user_preferences.usage_context != ""
        )
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            'state': self.conversation_state.value,
            'preferences': {
                'budget_range': f"₹{int(self.user_preferences.budget_min)} - ₹{int(self.user_preferences.budget_max)}" if self.user_preferences.budget_max < float('inf') else "Not specified",
                'colors': self.user_preferences.preferred_colors,
                'features': self.user_preferences.features_wanted,
                'usage': self.user_preferences.usage_context
            },
            'recommendations_shown': len(self.last_recommendations),
            'conversation_length': len(self.conversation_history)
        }
    
    def reset_conversation(self):
        """Reset conversation state for new user"""
        self.conversation_state = ConversationState.GREETING
        self.user_preferences = UserPreferences()
        self.conversation_history = []
        self.last_recommendations = []
        self.context_memory = {}

# Example usage and testing
if __name__ == "__main__":
    # This would be used with the recommendation engine
    print("ChatbotAgent class defined successfully!")
    print("Use this with the recommendation engine in the main Streamlit app.")