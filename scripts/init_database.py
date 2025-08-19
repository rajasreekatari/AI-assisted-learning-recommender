#!/usr/bin/env python3
"""
Database initialization script for AI Learning Recommender
Creates tables and seeds initial data in Snowflake
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.database import get_database_engine, test_connection
from models.database_models import Base
from sqlalchemy.orm import sessionmaker

def create_tables():
    """Create all database tables"""
    try:
        engine = get_database_engine()
        if not engine:
            print("❌ Failed to get database engine")
            return False
        
        print("🔨 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def seed_initial_data():
    """Seed initial data for skills taxonomy and career transitions"""
    try:
        engine = get_database_engine()
        if not engine:
            print("❌ Failed to get database engine")
            return False
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        print("🌱 Seeding initial data...")
        
        # Import models
        from models.database_models import Skill, CareerTransition
        
        # Seed skills taxonomy with explicit IDs
        skills_data = [
            # Programming Languages
            {"id": 1, "name": "Python", "category": "programming_languages", "difficulty_level": "Beginner", "estimated_learning_time": 80},
            {"id": 2, "name": "Java", "category": "programming_languages", "difficulty_level": "Intermediate", "estimated_learning_time": 120},
            {"id": 3, "name": "JavaScript", "category": "programming_languages", "difficulty_level": "Beginner", "estimated_learning_time": 60},
            {"id": 4, "name": "SQL", "category": "programming_languages", "difficulty_level": "Beginner", "estimated_learning_time": 40},
            {"id": 5, "name": "Go", "category": "programming_languages", "difficulty_level": "Intermediate", "estimated_learning_time": 100},
            {"id": 6, "name": "Rust", "category": "programming_languages", "difficulty_level": "Advanced", "estimated_learning_time": 200},
            
            # Frameworks
            {"id": 7, "name": "React", "category": "frameworks", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            {"id": 8, "name": "Django", "category": "frameworks", "difficulty_level": "Intermediate", "estimated_learning_time": 100},
            {"id": 9, "name": "Flask", "category": "frameworks", "difficulty_level": "Beginner", "estimated_learning_time": 60},
            {"id": 10, "name": "Spring", "category": "frameworks", "difficulty_level": "Advanced", "estimated_learning_time": 150},
            {"id": 11, "name": "Express", "category": "frameworks", "difficulty_level": "Intermediate", "estimated_learning_time": 70},
            
            # Databases
            {"id": 12, "name": "PostgreSQL", "category": "databases", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            {"id": 13, "name": "MongoDB", "category": "databases", "difficulty_level": "Beginner", "estimated_learning_time": 60},
            {"id": 14, "name": "Redis", "category": "databases", "difficulty_level": "Intermediate", "estimated_learning_time": 50},
            {"id": 15, "name": "Snowflake", "category": "databases", "difficulty_level": "Intermediate", "estimated_learning_time": 70},
            
            # Cloud Platforms
            {"id": 16, "name": "AWS", "category": "cloud_platforms", "difficulty_level": "Intermediate", "estimated_learning_time": 120},
            {"id": 17, "name": "Docker", "category": "cloud_platforms", "difficulty_level": "Beginner", "estimated_learning_time": 50},
            {"id": 18, "name": "Kubernetes", "category": "cloud_platforms", "difficulty_level": "Advanced", "estimated_learning_time": 150},
            
            # Data Engineering
            {"id": 19, "name": "Apache Spark", "category": "data_engineering", "difficulty_level": "Advanced", "estimated_learning_time": 120},
            {"id": 20, "name": "Airflow", "category": "data_engineering", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            {"id": 21, "name": "Pandas", "category": "data_engineering", "difficulty_level": "Beginner", "estimated_learning_time": 40},
            {"id": 22, "name": "NumPy", "category": "data_engineering", "difficulty_level": "Beginner", "estimated_learning_time": 30},
            
            # Machine Learning
            {"id": 23, "name": "scikit-learn", "category": "machine_learning", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            {"id": 24, "name": "TensorFlow", "category": "machine_learning", "difficulty_level": "Advanced", "estimated_learning_time": 150},
            {"id": 25, "name": "PyTorch", "category": "machine_learning", "difficulty_level": "Advanced", "estimated_learning_time": 150},
            {"id": 26, "name": "Hugging Face", "category": "machine_learning", "difficulty_level": "Intermediate", "estimated_learning_time": 60},
            {"id": 27, "name": "MLflow", "category": "machine_learning", "difficulty_level": "Intermediate", "estimated_learning_time": 40},
            
            # Additional Programming Languages
            {"id": 28, "name": "C++", "category": "programming_languages", "difficulty_level": "Advanced", "estimated_learning_time": 200},
            {"id": 29, "name": "C#", "category": "programming_languages", "difficulty_level": "Intermediate", "estimated_learning_time": 100},
            {"id": 30, "name": "TypeScript", "category": "programming_languages", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            
            # Additional Frameworks
            {"id": 31, "name": "Angular", "category": "frameworks", "difficulty_level": "Advanced", "estimated_learning_time": 120},
            {"id": 32, "name": "Vue.js", "category": "frameworks", "difficulty_level": "Intermediate", "estimated_learning_time": 70},
            {"id": 33, "name": "FastAPI", "category": "frameworks", "difficulty_level": "Intermediate", "estimated_learning_time": 60},
            
            # Additional Databases
            {"id": 34, "name": "BigQuery", "category": "databases", "difficulty_level": "Intermediate", "estimated_learning_time": 80},
            {"id": 35, "name": "DynamoDB", "category": "databases", "difficulty_level": "Intermediate", "estimated_learning_time": 70},
            
            # Additional Cloud Platforms
            {"id": 36, "name": "Azure", "category": "cloud_platforms", "difficulty_level": "Intermediate", "estimated_learning_time": 120},
            {"id": 37, "name": "GCP", "category": "cloud_platforms", "difficulty_level": "Intermediate", "estimated_learning_time": 120},
            {"id": 38, "name": "Terraform", "category": "cloud_platforms", "difficulty_level": "Advanced", "estimated_learning_time": 100},
            
            # Additional Data Engineering
            {"id": 39, "name": "Kafka", "category": "data_engineering", "difficulty_level": "Advanced", "estimated_learning_time": 100},
            {"id": 40, "name": "Hadoop", "category": "data_engineering", "difficulty_level": "Advanced", "estimated_learning_time": 120},
        ]
        
        # Add skills to database
        for skill_data in skills_data:
            skill = Skill(**skill_data)
            session.add(skill)
        
        # Seed career transitions with explicit IDs
        transitions_data = [
            {
                "id": 1,
                "from_role": "data_engineer",
                "to_role": "software_engineer",
                "difficulty_level": "Medium",
                "estimated_time": "6-12 months",
                "key_skills_to_learn": json.dumps(["Data Structures", "Algorithms", "System Design", "Software Architecture"]),
                "description": "Transition from data engineering to software engineering",
                "success_rate": 0.75
            },
            {
                "id": 2,
                "from_role": "frontend_developer",
                "to_role": "full_stack_developer",
                "difficulty_level": "Easy",
                "estimated_time": "4-8 months",
                "key_skills_to_learn": json.dumps(["Backend Development", "Database Design", "API Development", "DevOps Basics"]),
                "description": "Expand from frontend to full-stack development",
                "success_rate": 0.85
            },
            {
                "id": 3,
                "from_role": "software_engineer",
                "to_role": "ml_engineer",
                "difficulty_level": "Hard",
                "estimated_time": "8-16 months",
                "key_skills_to_learn": json.dumps(["Machine Learning", "Deep Learning", "Statistics", "MLOps"]),
                "description": "Transition from software engineering to ML engineering",
                "success_rate": 0.65
            },
            {
                "id": 4,
                "from_role": "junior_developer",
                "to_role": "senior_developer",
                "difficulty_level": "Medium",
                "estimated_time": "12-24 months",
                "key_skills_to_learn": json.dumps(["System Design", "Leadership", "Code Review", "Architecture Patterns", "Performance Optimization"]),
                "description": "Advance from junior to senior developer level",
                "success_rate": 0.70
            },
            {
                "id": 5,
                "from_role": "backend_developer",
                "to_role": "devops_engineer",
                "difficulty_level": "Medium",
                "estimated_time": "6-10 months",
                "key_skills_to_learn": json.dumps(["Docker", "Kubernetes", "CI/CD", "Infrastructure as Code", "Monitoring"]),
                "description": "Transition from backend development to DevOps",
                "success_rate": 0.80
            },
            {
                "id": 6,
                "from_role": "data_analyst",
                "to_role": "data_scientist",
                "difficulty_level": "Medium",
                "estimated_time": "8-14 months",
                "key_skills_to_learn": json.dumps(["Machine Learning", "Statistics", "Python", "Deep Learning", "Data Engineering"]),
                "description": "Advance from data analysis to data science",
                "success_rate": 0.75
            }
        ]
        
        # Add transitions to database
        for transition_data in transitions_data:
            transition = CareerTransition(**transition_data)
            session.add(transition)
        
        session.commit()
        print("✅ Initial data seeded successfully!")
        session.close()
        return True
        
    except Exception as e:
        print(f"❌ Error seeding data: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False

def main():
    """Main initialization function"""
    print("🚀 Initializing AI Learning Recommender Database...")
    print("=" * 60)
    
    # Test connection first
    if not test_connection():
        print("❌ Database connection failed. Please check your credentials.")
        return
    
    # Create tables
    if not create_tables():
        print("❌ Failed to create tables.")
        return
    
    # Seed initial data
    if not seed_initial_data():
        print("❌ Failed to seed initial data.")
        return
    
    print("=" * 60)
    print("🎉 Database initialization completed successfully!")
    print("✅ Tables created")
    print("✅ Initial data seeded")
    print("✅ Ready for use!")

if __name__ == "__main__":
    main()
