from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import database components
from services.database_service import db_service
from models.database_models import User, UserProfile as DBUserProfile, LearningPath as DBLearningPath, Skill, CareerTransition

def _extract_learning_resources(learning_plan: Dict[str, Any]) -> List[str]:
    """Extract learning resources from AI recommender response"""
    resources = []
    
    # Check if learning_resources is a dict with skill keys
    if isinstance(learning_plan.get('learning_resources'), dict):
        for skill, resource_data in learning_plan['learning_resources'].items():
            if isinstance(resource_data, dict):
                # Extract courses, books, practice, projects
                if 'courses' in resource_data:
                    resources.extend(resource_data['courses'][:2])  # Limit to 2 courses
                if 'books' in resource_data:
                    resources.extend(resource_data['books'][:1])   # Limit to 1 book
                if 'practice' in resource_data:
                    resources.extend(resource_data['practice'][:1]) # Limit to 1 practice
                if 'projects' in resource_data:
                    resources.extend(resource_data['projects'][:1]) # Limit to 1 project
    
    # If no structured resources, provide default ones
    if not resources:
        resources = [
            "Online courses on Coursera/Udemy",
            "Hands-on practice with real projects",
            "Industry-standard textbooks",
            "Open source contributions"
        ]
    
    return resources

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

# Pydantic models for API requests/responses
class UserProfileRequest(BaseModel):
    current_skills: List[str]
    learning_goals: Optional[List[str]] = None

class UserCreateRequest(BaseModel):
    username: str
    email: str
    experience_level: str
    target_role: str

class LearningPathResponse(BaseModel):
    path_id: int
    target_role: str
    skills_to_learn: List[str]
    estimated_duration: str
    learning_resources: List[str]
    ai_generated_plan: str
    status: str
    created_at: str

class SkillRecommendation(BaseModel):
    skill_name: str
    priority: str
    learning_resources: List[str]
    estimated_time: str

# Database service instance
# No more in-memory storage - everything goes to Snowflake!

@app.get("/")
async def root():
    return {"message": "AI-Assisted Learning Recommender API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "learning-recommender"}

@app.post("/user/create", response_model=dict)
async def create_user(user_data: UserCreateRequest):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = db_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")
        
        # Create user in database
        user = db_service.create_user(
            username=user_data.username,
            email=user_data.email,
            experience_level=user_data.experience_level,
            target_role=user_data.target_role
        )
        
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return {
            "user_id": user.id,
            "message": "User created successfully",
            "user": {
                "username": user.username,
                "email": user.email,
                "experience_level": user.experience_level,
                "target_role": user.target_role
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/profile/create", response_model=dict)
async def create_user_profile(profile: UserProfileRequest, user_id: int):
    """Create a new user profile"""
    try:
        # Check if user exists
        user = db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create user profile in database
        user_profile = db_service.create_user_profile(
            user_id=user_id,
            current_skills=profile.current_skills,
            learning_goals=profile.learning_goals
        )
        
        if not user_profile:
            raise HTTPException(status_code=500, detail="Failed to create user profile")
        
        return {
            "profile_id": user_profile.id,
            "message": "Profile created successfully",
            "profile": {
                "user_id": user_profile.user_id,
                "current_skills": profile.current_skills,
                "learning_goals": profile.learning_goals
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/profile/{profile_id}")
async def get_user_profile(profile_id: int):
    """Get user profile by ID"""
    try:
        profile = db_service.get_user_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {
            "profile_id": profile.id,
            "profile": {
                "user_id": profile.user_id,
                "current_skills": profile.current_skills,
                "learning_goals": profile.learning_goals,
                "created_at": profile.created_at.isoformat() if profile.created_at else None,
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommend/path", response_model=LearningPathResponse)
async def generate_learning_path(profile: UserProfileRequest, user_id: int):
    """Generate personalized learning path using AI and DSA components"""
    
    try:
        # Check if user exists
        user = db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Import AI recommender (DSA + Gen AI components)
        from models.ai_recommender import AIRecommender
        
        # Initialize AI recommender
        ai_recommender = AIRecommender()
        
        # Generate comprehensive learning plan
        user_profile = {
            "user_id": user_id,
            "skills": profile.current_skills
        }
        learning_plan = ai_recommender.generate_ai_plan(user_profile, user.target_role)
        
        # Create learning path in database
        db_learning_path = db_service.create_learning_path(
            user_id=user_id,
            target_role=user.target_role,
            skills_to_learn=learning_plan.get('learning_path', []),
            estimated_duration=f"{learning_plan.get('estimated_time', {}).get('total_months', 3)} months",
            ai_generated_plan=learning_plan.get('generation_method', 'AI plan generation failed'),
            learning_resources=_extract_learning_resources(learning_plan)
        )
        
        if not db_learning_path:
            raise HTTPException(status_code=500, detail="Failed to create learning path")
        
        # Return response
        return LearningPathResponse(
            path_id=db_learning_path.id,
            target_role=db_learning_path.target_role,
            skills_to_learn=learning_plan.get('learning_path', []),
            estimated_duration=f"{learning_plan.get('estimated_time', {}).get('total_months', 3)} months",
            learning_resources=_extract_learning_resources(learning_plan),
            ai_generated_plan=learning_plan.get('generation_method', 'AI plan generation failed'),
            status=db_learning_path.status,
            created_at=db_learning_path.created_at.isoformat() if db_learning_path.created_at else None
        )
        
    except Exception as e:
        # Fallback to mock response if AI fails
        print(f"AI recommendation failed: {str(e)}")
        
        # Create fallback learning path in database
        fallback_skills = ["Python", "SQL", "Data Analysis", "Machine Learning"]
        fallback_plan = "Begin with Python programming, then learn data manipulation with SQL and pandas. Progress to basic ML concepts."
        
        db_learning_path = db_service.create_learning_path(
            user_id=user_id,
            target_role=user.target_role,
            skills_to_learn=fallback_skills,
            estimated_duration="3-6 months",
            ai_generated_plan=fallback_plan,
            learning_resources=[
                "LeetCode for DSA",
                "Coursera/Udemy courses",
                "Real-world projects",
                "Open source contributions"
            ]
        )
        
        if not db_learning_path:
            raise HTTPException(status_code=500, detail="Failed to create fallback learning path")
        
        # Return fallback response
        return LearningPathResponse(
            path_id=db_learning_path.id,
            target_role=db_learning_path.target_role,
            skills_to_learn=fallback_skills,
            estimated_duration="3-6 months",
            learning_resources=[
                "LeetCode for DSA",
                "Coursera/Udemy courses",
                "Real-world projects",
                "Open source contributions"
            ],
            ai_generated_plan=fallback_plan,
            status=db_learning_path.status,
            created_at=db_learning_path.created_at.isoformat() if db_learning_path.created_at else None
        )

@app.get("/skills/tech-taxonomy")
async def get_tech_skills_taxonomy():
    """Get technology skills taxonomy for recommendations from database"""
    try:
        # Get all skills from database
        all_skills = db_service.get_all_skills()
        
        # Group skills by category
        taxonomy = {}
        for skill in all_skills:
            category = skill.category
            if category not in taxonomy:
                taxonomy[category] = []
            taxonomy[category].append(skill.name)
        
        return taxonomy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch skills taxonomy: {str(e)}")

@app.get("/career-paths")
async def get_career_transition_paths():
    """Get career transition paths from database"""
    try:
        # Get all career transitions from database
        transitions = db_service.get_career_transitions()
        
        # Format response
        career_paths = {}
        for transition in transitions:
            key = f"{transition.from_role}_to_{transition.to_role}"
            
            # Parse key_skills_to_learn from JSON string to list
            try:
                import json
                key_skills = json.loads(transition.key_skills_to_learn) if transition.key_skills_to_learn else []
            except (json.JSONDecodeError, TypeError):
                key_skills = []
            
            career_paths[key] = {
                "description": transition.description,
                "key_skills_to_learn": key_skills,
                "estimated_time": transition.estimated_time,
                "difficulty_level": transition.difficulty_level,
                "success_rate": transition.success_rate
            }
        
        return career_paths
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch career paths: {str(e)}")

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
        
        ai_recommender = AIRecommender()
        insights = ai_recommender.get_career_transition_path(from_role, to_role)
        
        return insights
        
    except Exception as e:
        return {"error": f"Failed to get career transition insights: {str(e)}"}

@app.get("/career/transitions/advanced")
async def get_advanced_career_transitions():
    """Get advanced career transition matrix with difficulty levels"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.skill_graph import SkillDependencyGraph
        
        skill_graph = SkillDependencyGraph()
        transitions = skill_graph.get_advanced_career_transitions()
        
        return {
            "message": "Advanced career transition matrix",
            "transitions": transitions,
            "total_paths": len(transitions)
        }
        
    except Exception as e:
        return {"error": f"Failed to get advanced career transitions: {str(e)}"}

@app.get("/career/transitions/recommendations/{current_role}")
async def get_transition_recommendations(current_role: str, experience_years: int = 2):
    """Get personalized transition recommendations based on experience"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from models.skill_graph import SkillDependencyGraph
        
        skill_graph = SkillDependencyGraph()
        recommendations = skill_graph.get_transition_recommendations(current_role, experience_years)
        
        return {
            "current_role": current_role,
            "experience_years": experience_years,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {"error": f"Failed to get transition recommendations: {str(e)}"}

@app.get("/ai/detailed-plan/{path_id}")
async def get_detailed_ai_plan(path_id: int):
    """Get detailed AI-generated learning plan from database"""
    try:
        # Get learning path from database
        learning_path = db_service.get_learning_path_by_id(path_id)
        if not learning_path:
            raise HTTPException(status_code=404, detail="Learning path not found")
        
        return {
            "path_id": learning_path.id,
            "target_role": learning_path.target_role,
            "skills_to_learn": learning_path.skills_to_learn,
            "estimated_duration": learning_path.estimated_duration,
            "ai_generated_plan": learning_path.ai_generated_plan,
            "learning_resources": learning_path.learning_resources,
            "status": learning_path.status,
            "created_at": learning_path.created_at.isoformat() if learning_path.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch learning path: {str(e)}")

# New database-powered endpoints
@app.get("/users/{user_id}/learning-paths")
async def get_user_learning_paths(user_id: int):
    """Get all learning paths for a specific user"""
    try:
        learning_paths = db_service.get_user_learning_paths(user_id)
        return {
            "user_id": user_id,
            "learning_paths": [
                {
                    "path_id": path.id,
                    "target_role": path.target_role,
                    "skills_to_learn": path.skills_to_learn,
                    "estimated_duration": path.estimated_duration,
                    "status": path.status,
                    "created_at": path.created_at.isoformat() if path.created_at else None
                }
                for path in learning_paths
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch learning paths: {str(e)}")

@app.get("/skills/category/{category}")
async def get_skills_by_category(category: str):
    """Get skills by specific category"""
    try:
        skills = db_service.get_skills_by_category(category)
        return {
            "category": category,
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "difficulty_level": skill.difficulty_level,
                    "estimated_learning_time": skill.estimated_learning_time
                }
                for skill in skills
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch skills by category: {str(e)}")

@app.get("/database/health")
async def database_health_check():
    """Check database connectivity"""
    try:
        # Test database connection
        from config.database import test_connection
        if test_connection():
            return {"status": "healthy", "database": "connected", "service": "learning-recommender"}
        else:
            return {"status": "unhealthy", "database": "disconnected", "service": "learning-recommender"}
    except Exception as e:
        return {"status": "error", "database": "error", "service": "learning-recommender", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
