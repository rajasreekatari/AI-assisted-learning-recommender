from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="AI-Assisted Learning Recommender",
    description="Personalized learning path recommender for tech professionals",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class UserProfile(BaseModel):
    current_skills: List[str]
    target_role: str
    experience_level: str
    learning_goals: Optional[List[str]] = None

class LearningPath(BaseModel):
    path_id: str
    target_role: str
    skills_to_learn: List[str]
    estimated_duration: str
    learning_resources: List[str]
    ai_generated_plan: str

class SkillRecommendation(BaseModel):
    skill_name: str
    priority: str
    learning_resources: List[str]
    estimated_time: str

# In-memory storage (replace with database later)
user_profiles = {}
learning_paths = {}

@app.get("/")
async def root():
    return {"message": "AI-Assisted Learning Recommender API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "learning-recommender"}

@app.post("/profile/create", response_model=dict)
async def create_user_profile(profile: UserProfile):
    """Create a new user profile"""
    profile_id = f"profile_{len(user_profiles) + 1}"
    user_profiles[profile_id] = profile.model_dump()
    
    return {
        "profile_id": profile_id,
        "message": "Profile created successfully",
        "profile": profile.model_dump()
    }

@app.get("/profile/{profile_id}")
async def get_user_profile(profile_id: str):
    """Get user profile by ID"""
    if profile_id not in user_profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return {"profile_id": profile_id, "profile": user_profiles[profile_id]}

@app.post("/recommend/path", response_model=LearningPath)
async def generate_learning_path(profile: UserProfile):
    """Generate personalized learning path using AI and DSA components"""
    
    try:
        # Import AI recommender (DSA + Gen AI components)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.ai_recommender import AIRecommender
        
        # Initialize AI recommender
        ai_recommender = AIRecommender()
        
        # Generate comprehensive learning plan
        user_profile = {
            "user_id": profile.user_id,
            "skills": profile.current_skills
        }
        learning_plan = ai_recommender.generate_ai_plan(user_profile, profile.target_role)
        
        # Create response
        path_id = f"path_{len(learning_paths) + 1}"
        
        learning_path = LearningPath(
            path_id=path_id,
            target_role=profile.target_role,
            skills_to_learn=learning_plan.get('target_skills', []),
            estimated_duration=f"{learning_plan.get('estimated_time', {}).get('total_months', 3)} months",
            learning_resources=learning_plan.get('learning_resources', []),
            ai_generated_plan=learning_plan.get('generation_method', 'AI plan generation failed')
        )
        
        learning_paths[path_id] = learning_path.model_dump()
        
        # Store full plan for detailed access
        learning_paths[f"detailed_{path_id}"] = learning_plan
        
        return learning_path
        
    except Exception as e:
        # Fallback to mock response if AI fails
        print(f"AI recommendation failed: {str(e)}")
        
        path_id = f"path_{len(learning_paths) + 1}"
        
        # Mock skill recommendations based on target role
        if "data engineer" in profile.target_role.lower():
            skills_to_learn = ["Apache Spark", "Airflow", "Snowflake", "Python", "SQL"]
            ai_plan = "Start with Python fundamentals, then learn SQL and data modeling. Progress to distributed computing with Spark and workflow orchestration with Airflow."
        elif "software engineer" in profile.target_role.lower():
            skills_to_learn = ["Data Structures", "Algorithms", "System Design", "Python/Java", "Databases"]
            ai_plan = "Focus on DSA fundamentals through LeetCode, then learn system design principles. Build full-stack projects to apply your knowledge."
        else:
            skills_to_learn = ["Python", "SQL", "Data Analysis", "Machine Learning"]
            ai_plan = "Begin with Python programming, then learn data manipulation with SQL and pandas. Progress to basic ML concepts."
        
        learning_path = LearningPath(
            path_id=path_id,
            target_role=profile.target_role,
            skills_to_learn=skills_to_learn,
            estimated_duration="3-6 months",
            learning_resources=[
                "LeetCode for DSA",
                "Coursera/Udemy courses",
                "Real-world projects",
                "Open source contributions"
            ],
            ai_generated_plan=ai_plan
        )
        
        learning_paths[path_id] = learning_path.model_dump()
        return learning_path

@app.get("/skills/tech-taxonomy")
async def get_tech_skills_taxonomy():
    """Get technology skills taxonomy for recommendations"""
    return {
        "programming_languages": [
            "Python", "Java", "JavaScript", "Go", "Rust", "C++", "C#"
        ],
        "frameworks": [
            "React", "Angular", "Vue.js", "Django", "Flask", "Spring", "Express"
        ],
        "databases": [
            "PostgreSQL", "MongoDB", "Redis", "Snowflake", "BigQuery", "DynamoDB"
        ],
        "cloud_platforms": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform"
        ],
        "data_engineering": [
            "Apache Spark", "Airflow", "Kafka", "Hadoop", "Pandas", "NumPy"
        ],
        "machine_learning": [
            "scikit-learn", "TensorFlow", "PyTorch", "Hugging Face", "MLflow"
        ]
    }

@app.get("/career-paths")
async def get_career_transition_paths():
    """Get predefined career transition paths"""
    return {
        "data_engineer_to_software_engineer": {
            "description": "Transition from DE to SDE",
            "key_skills_to_learn": [
                "Data Structures & Algorithms",
                "System Design",
                "Software Architecture",
                "Testing & Debugging"
            ],
            "estimated_time": "6-12 months"
        },
        "frontend_to_fullstack": {
            "description": "Expand from frontend to full-stack development",
            "key_skills_to_learn": [
                "Backend Development",
                "Database Design",
                "API Development",
                "DevOps Basics"
            ],
            "estimated_time": "4-8 months"
        },
        "junior_to_senior": {
            "description": "Advance from junior to senior level",
            "key_skills_to_learn": [
                "System Design",
                "Leadership",
                "Architecture Patterns",
                "Performance Optimization"
            ],
            "estimated_time": "12-18 months"
        }
    }

@app.get("/skills/graph/{skill_name}")
async def get_skill_dependencies(skill_name: str):
    """Get skill dependencies using DSA graph algorithms"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.skill_graph import SkillDependencyGraph
        
        skill_graph = SkillDependencyGraph()
        dependencies = skill_graph.get_skill_dependencies(skill_name)
        
        return dependencies
        
    except Exception as e:
        return {"error": f"Failed to get skill dependencies: {str(e)}"}

@app.get("/skills/path")
async def get_skill_learning_path(current_skills: str, target_skills: str):
    """Get learning path between skills using graph algorithms"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.skill_graph import SkillDependencyGraph
        
        # Parse skills from query parameters
        current_skills_list = [s.strip() for s in current_skills.split(',')]
        target_skills_list = [s.strip() for s in target_skills.split(',')]
        
        skill_graph = SkillDependencyGraph()
        path_result = skill_graph.get_skill_path(current_skills_list, target_skills_list)
        
        return path_result
        
    except Exception as e:
        return {"error": f"Failed to get skill path: {str(e)}"}

@app.get("/ai/career-transition/{from_role}/{to_role}")
async def get_ai_career_transition_insights(from_role: str, to_role: str):
    """Get AI-powered insights for career transitions"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.ai_recommender import AIRecommender
        
        ai_recommender = AIRecommender(use_local_model=True)
        insights = ai_recommender.get_career_transition_insights(from_role, to_role)
        
        return insights
        
    except Exception as e:
        return {"error": f"Failed to get career transition insights: {str(e)}"}

@app.get("/ai/detailed-plan/{path_id}")
async def get_detailed_ai_plan(path_id: str):
    """Get detailed AI-generated learning plan"""
    detailed_key = f"detailed_{path_id}"
    
    if detailed_key not in learning_paths:
        raise HTTPException(status_code=404, detail="Detailed plan not found")
    
    return learning_paths[detailed_key]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
