#!/usr/bin/env python3
"""
Test script for AI-Assisted Learning Recommender components
Tests DSA (NetworkX graphs) and Gen AI (HuggingFace) components
"""

import sys
import os
import json
from typing import Dict, Any

def test_skill_graph():
    """Test the DSA component: Skill dependency graph"""
    print("🧪 Testing Skill Dependency Graph (DSA Component)...")
    
    try:
        from models.skill_graph import SkillDependencyGraph
        
        # Initialize graph
        skill_graph = SkillDependencyGraph()
        print("✅ Skill graph initialized successfully")
        
        # Test skill path finding
        current_skills = ['python', 'sql']
        target_skills = ['django', 'postgresql']
        
        path_result = skill_graph.get_skill_path(current_skills, target_skills)
        print(f"✅ Skill path found: {len(path_result.get('missing_skills', []))} missing skills")
        
        # Test career transition
        career_path = skill_graph.get_career_transition_path('data_analyst', 'data_engineer')
        print(f"✅ Career transition path: {career_path.get('total_skills_to_learn', 0)} skills to learn")
        
        # Test skill dependencies
        deps = skill_graph.get_skill_dependencies('python')
        print(f"✅ Python dependencies: {len(deps.get('prerequisites', []))} prerequisites")
        
        print("🎉 Skill Graph (DSA Component) tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Skill Graph test failed: {str(e)}")
        return False

def test_ai_recommender():
    """Test the Gen AI component: AI-powered recommendations"""
    print("\n🧪 Testing AI Recommender (Gen AI Component)...")
    
    try:
        from models.ai_recommender import AIRecommender
        
        # Initialize AI recommender
        ai_recommender = AIRecommender()
        print("✅ AI recommender initialized successfully")
        
        # Test learning plan generation
        user_profile = {
            "user_id": "test_user",
            "skills": ["python", "sql"]
        }
        target_role = "data_engineer"
        
        learning_plan = ai_recommender.generate_ai_plan(user_profile, target_role)
        
        print(f"✅ Learning plan generated: {len(learning_plan.get('target_skills', []))} target skills")
        print(f"✅ Learning path: {len(learning_plan.get('learning_path', []))} skills to learn")
        
        # Test career transition path
        transition_path = ai_recommender.get_career_transition_path('software_engineer', 'data_engineer')
        print(f"✅ Career transition: {len(transition_path.get('transition_skills', []))} skills to learn")
        
        print("🎉 AI Recommender (Gen AI Component) tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ AI Recommender test failed: {str(e)}")
        return False

def test_backend_integration():
    """Test backend API integration"""
    print("\n🧪 Testing Backend Integration...")
    
    try:
        # Test if we can import the backend
        sys.path.append('backend')
        from main import app
        
        print("✅ Backend app imported successfully")
        
        # Test if we can import the models
        sys.path.append('models')
        from models.skill_graph import SkillDependencyGraph
        from models.ai_recommender import AIRecommender
        
        print("✅ Model components imported successfully")
        
        print("🎉 Backend Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backend Integration test failed: {str(e)}")
        return False

def test_data_processing():
    """Test data processing capabilities"""
    print("\n🧪 Testing Data Processing...")
    
    try:
        from data.skills_processor import TechSkillsProcessor
        
        # Initialize processor
        processor = TechSkillsProcessor()
        print("✅ Skills processor initialized successfully")
        
        # Test tech keywords
        tech_keywords = processor.tech_keywords
        print(f"✅ Tech keywords loaded: {len(tech_keywords)} categories")
        
        # Test tech job titles
        tech_jobs = processor.tech_job_titles
        print(f"✅ Tech job titles loaded: {len(tech_jobs)} roles")
        
        print("🎉 Data Processing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data Processing test failed: {str(e)}")
        return False

def run_performance_test():
    """Run basic performance tests"""
    print("\n🚀 Running Performance Tests...")
    
    try:
        from models.skill_graph import SkillDependencyGraph
        import time
        
        # Test graph initialization time
        start_time = time.time()
        skill_graph = SkillDependencyGraph()
        init_time = time.time() - start_time
        print(f"✅ Graph initialization: {init_time:.3f} seconds")
        
        # Test path finding performance
        start_time = time.time()
        for _ in range(10):
            skill_graph.get_skill_path(['python'], ['kubernetes'])
        path_time = time.time() - start_time
        print(f"✅ Path finding (10 iterations): {path_time:.3f} seconds")
        
        # Test AI recommendation performance
        from models.ai_recommender import AIRecommender
        
        start_time = time.time()
        ai_recommender = AIRecommender()
        ai_init_time = time.time() - start_time
        print(f"✅ AI model loading: {ai_init_time:.3f} seconds")
        
        print("🎉 Performance tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI-Assisted Learning Recommender - Component Testing")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Skill Graph (DSA)", test_skill_graph()))
    test_results.append(("AI Recommender (Gen AI)", test_ai_recommender()))
    test_results.append(("Backend Integration", test_backend_integration()))
    test_results.append(("Data Processing", test_data_processing()))
    test_results.append(("Performance", run_performance_test()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
