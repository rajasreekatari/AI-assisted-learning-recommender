import streamlit as st
import requests
import json
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="AI Learning Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_user(username: str, email: str, experience_level: str, target_role: str):
    """Create a user via API"""
    try:
        user_data = {
            "username": username,
            "email": email,
            "experience_level": experience_level,
            "target_role": target_role
        }
        
        response = requests.post(f"{API_BASE_URL}/user/create", json=user_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error creating user: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def create_user_profile(user_id: int, current_skills: List[str], learning_goals: List[str] = None):
    """Create a user profile via API"""
    try:
        profile_data = {
            "current_skills": current_skills,
            "learning_goals": learning_goals or []
        }
        
        response = requests.post(f"{API_BASE_URL}/profile/create?user_id={user_id}", json=profile_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error creating profile: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_learning_path(user_id: int, profile_data: Dict[str, Any]):
    """Get learning path recommendations via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/recommend/path?user_id={user_id}", json=profile_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting recommendations: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_tech_taxonomy():
    """Get technology skills taxonomy"""
    try:
        response = requests.get(f"{API_BASE_URL}/skills/tech-taxonomy")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_career_paths():
    """Get predefined career transition paths"""
    try:
        response = requests.get(f"{API_BASE_URL}/career-paths")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Main application
def main():
    st.title("🎓 AI-Assisted Learning Recommender")
    st.markdown("**Personalized learning paths for tech professionals navigating career transitions**")
    
    # Check API health
    if not check_api_health():
        st.warning("⚠️ FastAPI backend is not running. Please start the backend server first.")
        st.info("Run: `uvicorn backend.main:app --reload`")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "👤 Create Profile", "🎯 Get Recommendations", "🛠️ Tech Skills", "🚀 Career Paths"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 Create Profile":
        show_create_profile_page()
    elif page == "🎯 Get Recommendations":
        show_recommendations_page()
    elif page == "🛠️ Tech Skills":
        show_tech_skills_page()
    elif page == "🚀 Career Paths":
        show_career_paths_page()

def show_home_page():
    st.header("Welcome to Your AI Learning Assistant! 🚀")
    
    st.markdown("""
    This AI-powered system helps tech professionals like you:
    
    - **🔍 Identify skill gaps** for your target role
    - **📚 Generate personalized learning paths** using AI
    - **🎯 Navigate career transitions** (DE → SDE, Frontend → Full Stack, etc.)
    - **⚡ Get curated resources** and learning materials
    
    ### How it works:
    1. **Create your profile** with current skills and goals
    2. **Get AI-generated recommendations** for your learning path
    3. **Follow personalized guidance** to reach your career goals
    
    ### Perfect for:
    - Data Engineers transitioning to Software Engineering
    - Frontend developers becoming Full Stack engineers
    - Junior developers advancing to senior roles
    - Career changers entering tech
    """)
    
    # Quick start section
    st.subheader("🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Step 1:** Go to 'Create Profile' and tell us about your skills and goals")
    
    with col2:
        st.info("**Step 2:** Get personalized learning recommendations in 'Get Recommendations'")

def show_create_profile_page():
    st.header("👤 Create Your Learning Profile")
    
    with st.form("user_profile_form"):
        st.subheader("Tell us about yourself")
        
        # User information
        username = st.text_input(
            "**Username:**",
            placeholder="e.g., john_doe",
            help="Choose a unique username for your account"
        )
        
        email = st.text_input(
            "**Email:**",
            placeholder="e.g., john@example.com",
            help="Your email address"
        )
        
        # Current skills input
        st.write("**What skills do you currently have?**")
        current_skills = st.text_area(
            "Enter your current skills (one per line or comma-separated):",
            placeholder="Python, SQL, Data Analysis, Machine Learning...",
            help="List your current technical skills, programming languages, tools, etc."
        )
        
        # Target role
        target_role = st.text_input(
            "**What role are you targeting?**",
            placeholder="e.g., Software Engineer, Data Scientist, Full Stack Developer...",
            help="The role you want to transition into or advance towards"
        )
        
        # Experience level
        experience_level = st.selectbox(
            "**What's your current experience level?**",
            ["Beginner", "Intermediate", "Advanced", "Expert"]
        )
        
        # Learning goals
        learning_goals = st.text_area(
            "**What are your learning goals? (Optional)**",
            placeholder="e.g., Learn system design, Master DSA, Build full-stack apps...",
            help="Specific skills or knowledge areas you want to focus on"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Create Profile & Get Recommendations")
        
        if submitted:
            if not username or not email or not current_skills or not target_role:
                st.error("Please fill in all required fields!")
                return
            
            # Process skills input
            skills_list = [skill.strip() for skill in current_skills.replace('\n', ',').split(',') if skill.strip()]
            goals_list = [goal.strip() for goal in learning_goals.replace('\n', ',').split(',') if goal.strip()] if learning_goals else []
            
            # Create user first
            with st.spinner("Creating your account..."):
                user_result = create_user(username, email, experience_level, target_role)
            
            if user_result:
                user_id = user_result['user_id']
                st.success("✅ Account created successfully!")
                
                # Create user profile
                with st.spinner("Creating your learning profile..."):
                    profile_result = create_user_profile(user_id, skills_list, goals_list)
                
                if profile_result:
                    st.success("✅ Learning profile created successfully!")
                    st.session_state['user_id'] = user_id
                    st.session_state['user_profile'] = profile_result
                    st.session_state['profile_data'] = {
                        "current_skills": skills_list,
                        "target_role": target_role,
                        "experience_level": experience_level,
                        "learning_goals": goals_list
                    }
                    
                    # Auto-navigate to recommendations
                    st.info("🎯 Now let's get your personalized learning path!")
                    st.balloons()
                else:
                    st.error("Failed to create learning profile. Please try again.")
            else:
                st.error("Failed to create account. Please try again.")

def show_recommendations_page():
    st.header("🎯 Your Personalized Learning Path")
    
    if 'profile_data' not in st.session_state:
        st.warning("Please create a profile first!")
        st.info("Go to 'Create Profile' to get started.")
        return
    
    profile_data = st.session_state['profile_data']
    
    # Display current profile
    st.subheader("📋 Your Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Skills:**")
        for skill in profile_data['current_skills']:
            st.write(f"• {skill}")
    
    with col2:
        st.write("**Target Role:**", profile_data['target_role'])
        st.write("**Experience Level:**", profile_data['experience_level'])
        if profile_data['learning_goals']:
            st.write("**Learning Goals:**")
            for goal in profile_data['learning_goals']:
                st.write(f"• {goal}")
    
    # Get recommendations
    if st.button("🔄 Get Updated Recommendations"):
        if 'user_id' not in st.session_state:
            st.error("User ID not found. Please create a profile first.")
            return
            
        with st.spinner("Generating your personalized learning path..."):
            recommendations = get_learning_path(st.session_state['user_id'], profile_data)
        
        if recommendations:
            st.session_state['recommendations'] = recommendations
            display_recommendations(recommendations)
    
    # Display existing recommendations
    elif 'recommendations' in st.session_state:
        display_recommendations(st.session_state['recommendations'])

def display_recommendations(recommendations: Dict[str, Any]):
    st.subheader("🚀 Your AI-Generated Learning Path")
    
    # Path overview
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Target Role", recommendations['target_role'])
    with col2:
        st.metric("Estimated Duration", recommendations['estimated_duration'])
    
    # Skills to learn
    st.write("**📚 Skills to Learn:**")
    for i, skill in enumerate(recommendations['skills_to_learn'], 1):
        st.write(f"{i}. **{skill}**")
    
    # AI-generated plan
    st.write("**🤖 AI-Generated Learning Plan:**")
    st.info(recommendations['ai_generated_plan'])
    
    # Learning resources
    st.write("**📖 Recommended Learning Resources:**")
    for resource in recommendations['learning_resources']:
        st.write(f"• {resource}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Plan"):
            st.success("Plan saved! (Feature coming soon)")
    
    with col2:
        if st.button("📅 Schedule Learning"):
            st.info("Calendar integration coming soon!")
    
    with col3:
        if st.button("🔄 New Recommendations"):
            st.session_state.pop('recommendations', None)
            st.rerun()

def show_tech_skills_page():
    st.header("🛠️ Technology Skills Taxonomy")
    
    with st.spinner("Loading skills taxonomy..."):
        taxonomy = get_tech_taxonomy()
    
    if not taxonomy:
        st.error("Could not fetch skills taxonomy. Please check if the backend is running.")
        return
    

    
    # Display skills by category
    for category, skills in taxonomy.items():
        st.subheader(f"**{category.replace('_', ' ').title()}**")
        
        # Check if skills is a list
        if isinstance(skills, list):
            # Create columns for better layout
            cols = st.columns(3)
            for i, skill in enumerate(skills):
                col_idx = i % 3
                with cols[col_idx]:
                    st.write(f"• {skill}")
        else:
            st.write(f"⚠️ Skills for {category} is not a list: {type(skills)}")
            st.write(f"Content: {skills}")
        
        st.divider()

def show_career_paths_page():
    st.header("🚀 Career Transition Paths")
    
    with st.spinner("Loading career paths..."):
        career_paths = get_career_paths()
    
    if not career_paths:
        st.error("Could not fetch career paths. Please check if the backend is running.")
        return
    

    
    # Display each career path
    for path_key, path_info in career_paths.items():
        with st.expander(f"**{path_key.replace('_', ' ').title()}**"):
            st.write(f"**Description:** {path_info['description']}")
            st.write(f"**Estimated Time:** {path_info['estimated_time']}")
            
            st.write("**Key Skills to Learn:**")
            if isinstance(path_info['key_skills_to_learn'], list):
                for skill in path_info['key_skills_to_learn']:
                    st.write(f"• {skill}")
            else:
                st.write(f"⚠️ Key skills is not a list: {path_info['key_skills_to_learn']}")
            
            st.divider()

if __name__ == "__main__":
    main()

