from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Table, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Association table for many-to-many relationships
user_skills = Table(
    'user_skills',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('skill_id', Integer, ForeignKey('skills.id'))
)

class User(Base):
    """User profile model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    experience_level = Column(String(50), nullable=False)  # Beginner, Intermediate, Advanced, Expert
    target_role = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    skills = relationship("Skill", secondary=user_skills, back_populates="users")
    learning_paths = relationship("LearningPath", back_populates="user")
    profiles = relationship("UserProfile", back_populates="user")

class Skill(Base):
    """Skills taxonomy model"""
    __tablename__ = 'skills'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(100), nullable=False)  # programming_languages, frameworks, databases, etc.
    difficulty_level = Column(String(50), nullable=False)  # Beginner, Intermediate, Advanced
    description = Column(Text, nullable=True)
    estimated_learning_time = Column(Integer, nullable=True)  # in hours
    
    # Relationships
    users = relationship("User", secondary=user_skills, back_populates="skills")

class SkillPrerequisite(Base):
    """Skill dependency relationships"""
    __tablename__ = 'skill_prerequisites'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(Integer, ForeignKey('skills.id'), nullable=False)
    prerequisite_id = Column(Integer, ForeignKey('skills.id'), nullable=False)
    dependency_strength = Column(Float, default=1.0)  # How strong the dependency is (0.0 to 1.0)
    
    # Relationships
    skill = relationship("Skill", foreign_keys=[skill_id])
    prerequisite = relationship("Skill", foreign_keys=[prerequisite_id])

class UserProfile(Base):
    """User profile with current skills and goals"""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    current_skills = Column(Text, nullable=False)  # JSON string of current skills
    learning_goals = Column(Text, nullable=True)  # JSON string of learning goals
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="profiles")

class LearningPath(Base):
    """AI-generated learning paths"""
    __tablename__ = 'learning_paths'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    target_role = Column(String(100), nullable=False)
    skills_to_learn = Column(Text, nullable=False)  # JSON string of skills to learn
    estimated_duration = Column(String(100), nullable=False)  # e.g., "3-6 months"
    ai_generated_plan = Column(Text, nullable=False)
    learning_resources = Column(Text, nullable=True)  # JSON string of resources
    status = Column(String(50), default="active")  # active, completed, paused
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="learning_paths")

class CareerTransition(Base):
    """Career transition paths and requirements"""
    __tablename__ = 'career_transitions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_role = Column(String(100), nullable=False)
    to_role = Column(String(100), nullable=False)
    difficulty_level = Column(String(50), nullable=False)  # Easy, Medium, Hard
    estimated_time = Column(String(100), nullable=False)  # e.g., "6-12 months"
    key_skills_to_learn = Column(Text, nullable=False)  # JSON string of skills
    description = Column(Text, nullable=True)
    success_rate = Column(Float, nullable=True)  # Historical success rate (0.0 to 1.0)
    
    __table_args__ = (
        # Ensure unique combinations of from_role and to_role
        # Note: Snowflake doesn't support sqlite_on_conflict, using unique constraint instead
    )

class JobData(Base):
    """Processed job posting data"""
    __tablename__ = 'job_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_title = Column(String(200), nullable=False)
    company = Column(String(200), nullable=True)
    location = Column(String(200), nullable=True)
    required_skills = Column(Text, nullable=True)  # JSON string of required skills
    preferred_skills = Column(Text, nullable=True)  # JSON string of preferred skills
    experience_level = Column(String(100), nullable=True)
    salary_range = Column(String(100), nullable=True)
    source = Column(String(100), nullable=False)  # LinkedIn, GitHub Jobs, etc.
    posted_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
