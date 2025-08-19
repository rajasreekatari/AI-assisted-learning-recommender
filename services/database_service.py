"""
Database service layer for AI Learning Recommender
Handles all database operations using SQLAlchemy ORM
"""

import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from config.database import get_database_session
from models.database_models import User, Skill, UserProfile, LearningPath, CareerTransition, JobData

class DatabaseService:
    """Service class for database operations"""
    
    def __init__(self):
        self.session: Optional[Session] = None
    
    def __enter__(self):
        self.session = get_database_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    def get_session(self) -> Session:
        """Get database session"""
        if not self.session:
            self.session = get_database_session()
        return self.session
    
    # User operations
    def create_user(self, username: str, email: str, experience_level: str, target_role: str) -> Optional[User]:
        """Create a new user"""
        try:
            session = self.get_session()
            
            # Get the next available ID
            max_id = session.query(User.id).order_by(User.id.desc()).first()
            next_id = 1 if max_id is None else max_id[0] + 1
            
            user = User(
                id=next_id,
                username=username,
                email=email,
                experience_level=experience_level,
                target_role=target_role
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        except Exception as e:
            print(f"Error creating user: {e}")
            if session:
                session.rollback()
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            session = self.get_session()
            return session.query(User).filter(User.id == user_id).first()
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            session = self.get_session()
            return session.query(User).filter(User.email == email).first()
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    # Skills operations
    def get_all_skills(self) -> List[Skill]:
        """Get all skills"""
        try:
            session = self.get_session()
            return session.query(Skill).all()
        except Exception as e:
            print(f"Error getting skills: {e}")
            return []
    
    def get_skills_by_category(self, category: str) -> List[Skill]:
        """Get skills by category"""
        try:
            session = self.get_session()
            return session.query(Skill).filter(Skill.category == category).all()
        except Exception as e:
            print(f"Error getting skills by category: {e}")
            return []
    
    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        """Get skill by name"""
        try:
            session = self.get_session()
            return session.query(Skill).filter(Skill.name == name).first()
        except Exception as e:
            print(f"Error getting skill by name: {e}")
            return None
    
    # User Profile operations
    def create_user_profile(self, user_id: int, current_skills: List[str], learning_goals: List[str] = None) -> Optional[UserProfile]:
        """Create user profile"""
        try:
            session = self.get_session()
            
            # Get the next available ID
            max_id = session.query(UserProfile.id).order_by(UserProfile.id.desc()).first()
            next_id = 1 if max_id is None else max_id[0] + 1
            
            profile = UserProfile(
                id=next_id,
                user_id=user_id,
                current_skills=json.dumps(current_skills),
                learning_goals=json.dumps(learning_goals) if learning_goals else None
            )
            session.add(profile)
            session.commit()
            session.refresh(profile)
            return profile
        except Exception as e:
            print(f"Error creating user profile: {e}")
            if session:
                session.rollback()
            return None
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """Get user profile"""
        try:
            session = self.get_session()
            return session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    # Learning Path operations
    def create_learning_path(self, user_id: int, target_role: str, skills_to_learn: List[str], 
                           estimated_duration: str, ai_generated_plan: str, 
                           learning_resources: List[str] = None) -> Optional[LearningPath]:
        """Create learning path"""
        try:
            session = self.get_session()
            
            # Get the next available ID
            max_id = session.query(LearningPath.id).order_by(LearningPath.id.desc()).first()
            next_id = 1 if max_id is None else max_id[0] + 1
            
            path = LearningPath(
                id=next_id,
                user_id=user_id,
                target_role=target_role,
                skills_to_learn=json.dumps(skills_to_learn),
                estimated_duration=estimated_duration,
                ai_generated_plan=ai_generated_plan,
                learning_resources=json.dumps(learning_resources) if learning_resources else None
            )
            session.add(path)
            session.commit()
            session.refresh(path)
            return path
        except Exception as e:
            print(f"Error creating learning path: {e}")
            if session:
                session.rollback()
            return None
    
    def get_user_learning_paths(self, user_id: int) -> List[LearningPath]:
        """Get all learning paths for a user"""
        try:
            session = self.get_session()
            return session.query(LearningPath).filter(LearningPath.user_id == user_id).all()
        except Exception as e:
            print(f"Error getting learning paths: {e}")
            return []
    
    def get_learning_path_by_id(self, path_id: int) -> Optional[LearningPath]:
        """Get learning path by ID"""
        try:
            session = self.get_session()
            return session.query(LearningPath).filter(LearningPath.id == path_id).first()
        except Exception as e:
            print(f"Error getting learning path by ID: {e}")
            return None
    
    # Career Transition operations
    def get_career_transitions(self) -> List[CareerTransition]:
        """Get all career transitions"""
        try:
            session = self.get_session()
            return session.query(CareerTransition).all()
        except Exception as e:
            print(f"Error getting career transitions: {e}")
            return []
    
    def get_career_transition(self, from_role: str, to_role: str) -> Optional[CareerTransition]:
        """Get specific career transition"""
        try:
            session = self.get_session()
            return session.query(CareerTransition).filter(
                and_(
                    CareerTransition.from_role == from_role,
                    CareerTransition.to_role == to_role
                )
            ).first()
        except Exception as e:
            print(f"Error getting career transition: {e}")
            return None
    
    def get_transitions_by_difficulty(self, difficulty: str) -> List[CareerTransition]:
        """Get career transitions by difficulty level"""
        try:
            session = self.get_session()
            return session.query(CareerTransition).filter(CareerTransition.difficulty_level == difficulty).all()
        except Exception as e:
            print(f"Error getting transitions by difficulty: {e}")
            return []
    
    # Job Data operations
    def create_job_data(self, job_title: str, company: str, location: str, 
                       required_skills: List[str], preferred_skills: List[str] = None,
                       experience_level: str = None, salary_range: str = None,
                       source: str = "LinkedIn") -> Optional[JobData]:
        """Create job data entry"""
        try:
            session = self.get_session()
            job = JobData(
                job_title=job_title,
                company=company,
                location=location,
                required_skills=json.dumps(required_skills),
                preferred_skills=json.dumps(preferred_skills) if preferred_skills else None,
                experience_level=experience_level,
                salary_range=salary_range,
                source=source
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
        except Exception as e:
            print(f"Error creating job data: {e}")
            if session:
                session.rollback()
            return None
    
    def search_jobs_by_skills(self, skills: List[str], limit: int = 10) -> List[JobData]:
        """Search jobs by required skills"""
        try:
            session = self.get_session()
            # This is a simplified search - in production you'd want more sophisticated text search
            jobs = session.query(JobData).limit(limit).all()
            # Filter jobs that match the skills (simplified logic)
            matching_jobs = []
            for job in jobs:
                if job.required_skills:
                    job_skills = json.loads(job.required_skills)
                    if any(skill.lower() in [js.lower() for js in job_skills] for skill in skills):
                        matching_jobs.append(job)
            return matching_jobs
        except Exception as e:
            print(f"Error searching jobs by skills: {e}")
            return []

# Global database service instance
db_service = DatabaseService()
