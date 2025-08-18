import json
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .skill_graph import SkillDependencyGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIRecommender:
    """
    Gen AI component for generating personalized learning plans.
    Uses HuggingFace transformers for text generation and integrates with skill graphs.
    """
    
    def __init__(self, model_name: str = "facebook/bart-base"):
        """
        Initialize the AI recommender with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info(f"Loaded AI model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load AI model {model_name}: {e}")
            logger.info("Falling back to rule-based recommendations")
            self.model = None
            self.tokenizer = None
        
        self.skill_graph = SkillDependencyGraph()
        
        # Career transition mappings
        self.career_transitions = {
            "data_engineer": {
                "software_engineer": ["python", "java", "system_design", "algorithms"],
                "ml_engineer": ["machine_learning", "deep_learning", "mlops", "python"],
                "data_scientist": ["statistics", "machine_learning", "python", "r"]
            },
            "software_engineer": {
                "data_engineer": ["sql", "data_warehousing", "etl", "python"],
                "ml_engineer": ["machine_learning", "python", "statistics", "mlops"],
                "devops_engineer": ["docker", "kubernetes", "ci_cd", "infrastructure"]
            },
            "frontend_developer": {
                "full_stack_developer": ["backend_development", "databases", "api_design", "python"],
                "ui_ux_designer": ["design_principles", "user_research", "prototyping", "figma"]
            }
        }
    
    def get_target_skills_for_role(self, target_role: str) -> List[str]:
        """
        Get the key skills required for a target role.
        
        Args:
            target_role: The target career role
            
        Returns:
            List of required skills
        """
        role_skill_mappings = {
            "data_engineer": ["sql", "python", "etl", "data_warehousing", "spark", "hadoop"],
            "software_engineer": ["python", "java", "algorithms", "system_design", "databases"],
            "ml_engineer": ["python", "machine_learning", "deep_learning", "statistics", "mlops"],
            "data_scientist": ["python", "statistics", "machine_learning", "sql", "r"],
            "devops_engineer": ["docker", "kubernetes", "ci_cd", "infrastructure", "monitoring"],
            "full_stack_developer": ["frontend", "backend", "databases", "deployment", "api_design"]
        }
        
        return role_skill_mappings.get(target_role.lower(), [])
    
    def generate_ai_plan(self, user_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """
        Generate a personalized learning plan using AI.
        
        Args:
            user_profile: User's current skills and background
            target_role: Target career role
            
        Returns:
            Generated learning plan
        """
        try:
            if self.model and self.tokenizer:
                return self._generate_with_ai(user_profile, target_role)
            else:
                return self._generate_rule_based(user_profile, target_role)
        except Exception as e:
            logger.error(f"Error generating AI plan: {e}")
            return self._generate_rule_based(user_profile, target_role)
    
    def _generate_with_ai(self, user_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """
        Generate plan using the AI model.
        """
        current_skills = user_profile.get("skills", [])
        target_skills = self.get_target_skills_for_role(target_role)
        
        # Create input text for the model
        input_text = f"Current skills: {', '.join(current_skills)}. Target role: {target_role}. Required skills: {', '.join(target_skills)}"
        
        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the generated text into structured format
        return self._parse_ai_output(generated_text, user_profile, target_role)
    
    def _generate_rule_based(self, user_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """
        Generate plan using rule-based logic when AI is not available.
        """
        current_skills = user_profile.get("skills", [])
        target_skills = self.get_target_skills_for_role(target_role)
        
        # Find skill gaps
        skill_gaps = [skill for skill in target_skills if skill not in current_skills]
        
        # Get learning path from skill graph
        learning_path = []
        for skill in skill_gaps:
            # For single skill, we need to provide current skills and target skills
            path_result = self.skill_graph.get_skill_path(current_skills, [skill])
            if path_result and 'learning_paths' in path_result:
                # Extract the path from the result
                skill_path = path_result['learning_paths'].get(skill, {})
                if 'path' in skill_path:
                    learning_path.extend(skill_path['path'])
        
        # Remove duplicates and create stages
        unique_path = list(dict.fromkeys(learning_path))  # Preserve order
        stages = self._create_learning_stages(unique_path)
        
        return {
            "user_id": user_profile.get("user_id"),
            "target_role": target_role,
            "current_skills": current_skills,
            "target_skills": target_skills,
            "skill_gaps": skill_gaps,
            "learning_path": unique_path,
            "stages": stages,
            "estimated_time": self._estimate_total_time(unique_path),
            "success_metrics": self._define_success_metrics(target_role),
            "generation_method": "rule_based"
        }
    
    def _parse_ai_output(self, ai_text: str, user_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """
        Parse AI-generated text into structured format.
        Falls back to rule-based if parsing fails.
        """
        try:
            # Simple parsing - extract skills mentioned
            current_skills = user_profile.get("skills", [])
            target_skills = self.get_target_skills_for_role(target_role)
            
            # Extract skills mentioned in AI text
            mentioned_skills = []
            for skill in target_skills:
                if skill.lower() in ai_text.lower():
                    mentioned_skills.append(skill)
            
            # If AI didn't provide enough structure, fall back to rule-based
            if len(mentioned_skills) < len(target_skills) * 0.5:
                return self._generate_rule_based(user_profile, target_role)
            
            # Create structured plan from AI output
            skill_gaps = [skill for skill in target_skills if skill not in current_skills]
            learning_path = self._create_ai_learning_path(ai_text, skill_gaps)
            stages = self._create_learning_stages(learning_path)
            
            return {
                "user_id": user_profile.get("user_id"),
                "target_role": target_role,
                "current_skills": current_skills,
                "target_skills": target_skills,
                "skill_gaps": skill_gaps,
                "learning_path": learning_path,
                "stages": stages,
                "estimated_time": self._estimate_total_time(learning_path),
                "success_metrics": self._define_success_metrics(target_role),
                "generation_method": "ai_generated",
                "ai_explanation": ai_text[:500] + "..." if len(ai_text) > 500 else ai_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing AI output: {e}")
            return self._generate_rule_based(user_profile, target_role)
    
    def _create_ai_learning_path(self, ai_text: str, skill_gaps: List[str]) -> List[str]:
        """
        Create learning path from AI-generated text.
        """
        # Simple extraction - look for skill mentions in order
        learning_path = []
        for skill in skill_gaps:
            if skill.lower() in ai_text.lower():
                learning_path.append(skill)
        
        # If AI didn't provide clear path, use skill graph
        if len(learning_path) < len(skill_gaps) * 0.7:
            for skill in skill_gaps:
                path_result = self.skill_graph.get_skill_path(current_skills, [skill])
                if path_result and 'learning_paths' in path_result:
                    skill_path = path_result['learning_paths'].get(skill, {})
                    if 'path' in skill_path:
                        learning_path.extend(skill_path['path'])
        
        return list(dict.fromkeys(learning_path))  # Remove duplicates, preserve order
    
    def enhance_plan_with_resources(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the learning plan with specific resources and learning materials.
        """
        enhanced_plan = plan.copy()
        
        # Add learning resources for each skill
        learning_resources = {}
        for skill in plan.get("learning_path", []):
            resources = self._get_learning_resources(skill)
            learning_resources[skill] = resources
        
        enhanced_plan["learning_resources"] = learning_resources
        
        # Add practice projects
        enhanced_plan["practice_projects"] = self._get_practice_projects(plan["target_role"])
        
        return enhanced_plan
    
    def _get_learning_resources(self, skill: str) -> Dict[str, Any]:
        """
        Get learning resources for a specific skill.
        """
        resource_mappings = {
            "python": {
                "courses": ["Python for Everybody (Coursera)", "Complete Python Bootcamp (Udemy)"],
                "books": ["Python Crash Course", "Fluent Python"],
                "practice": ["LeetCode Python problems", "HackerRank Python track"],
                "projects": ["Build a web scraper", "Create a data analysis tool"]
            },
            "sql": {
                "courses": ["SQL for Data Science (Coursera)", "Complete SQL Bootcamp (Udemy)"],
                "books": ["SQL Cookbook", "Learning SQL"],
                "practice": ["HackerRank SQL track", "SQLZoo exercises"],
                "projects": ["Design a database schema", "Build reporting queries"]
            },
            "machine_learning": {
                "courses": ["Machine Learning (Coursera)", "Deep Learning Specialization"],
                "books": ["Hands-On Machine Learning", "Pattern Recognition"],
                "practice": ["Kaggle competitions", "Scikit-learn tutorials"],
                "projects": ["Predict house prices", "Image classification model"]
            }
        }
        
        return resource_mappings.get(skill.lower(), {
            "courses": ["Online courses on Coursera/Udemy"],
            "books": ["Industry-standard textbooks"],
            "practice": ["Hands-on exercises and problems"],
            "projects": ["Build real-world applications"]
        })
    
    def _get_practice_projects(self, target_role: str) -> List[Dict[str, str]]:
        """
        Get practice projects for the target role.
        """
        project_mappings = {
            "data_engineer": [
                {"name": "ETL Pipeline", "description": "Build a data pipeline using Python and SQL"},
                {"name": "Data Warehouse", "description": "Design and implement a data warehouse"},
                {"name": "Real-time Streaming", "description": "Create a real-time data processing system"}
            ],
            "software_engineer": [
                {"name": "Web Application", "description": "Build a full-stack web application"},
                {"name": "API Service", "description": "Create a RESTful API with authentication"},
                {"name": "Microservice", "description": "Design and implement a microservice architecture"}
            ],
            "ml_engineer": [
                {"name": "ML Pipeline", "description": "Build an end-to-end ML pipeline"},
                {"name": "Model Deployment", "description": "Deploy ML models to production"},
                {"name": "A/B Testing", "description": "Implement A/B testing for ML models"}
            ]
        }
        
        return project_mappings.get(target_role.lower(), [
            {"name": "Portfolio Project", "description": "Build a project showcasing your skills"},
            {"name": "Open Source Contribution", "description": "Contribute to open source projects"}
        ])
    
    def _create_learning_stages(self, learning_path: List[str]) -> List[Dict[str, Any]]:
        """
        Create structured learning stages from the learning path.
        """
        if not learning_path:
            return []
        
        # Group skills into logical stages
        stages = []
        skills_per_stage = max(1, len(learning_path) // 4)  # 4 stages max
        
        for i in range(0, len(learning_path), skills_per_stage):
            stage_skills = learning_path[i:i + skills_per_stage]
            stage = {
                "stage_number": len(stages) + 1,
                "skills": stage_skills,
                "estimated_weeks": len(stage_skills) * 2,  # 2 weeks per skill
                "milestones": [f"Complete {skill} fundamentals" for skill in stage_skills]
            }
            stages.append(stage)
        
        return stages
    
    def _estimate_total_time(self, learning_path: List[str]) -> Dict[str, Any]:
        """
        Estimate total time required for the learning path.
        """
        total_weeks = len(learning_path) * 2  # 2 weeks per skill
        
        return {
            "total_weeks": total_weeks,
            "total_months": round(total_weeks / 4, 1),
            "study_hours_per_week": 10,
            "total_study_hours": total_weeks * 10
        }
    
    def _define_success_metrics(self, target_role: str) -> List[str]:
        """
        Define success metrics for the target role.
        """
        base_metrics = [
            "Complete all learning stages",
            "Build portfolio projects",
            "Pass technical assessments",
            "Network with professionals in the field"
        ]
        
        role_specific_metrics = {
            "data_engineer": [
                "Design and implement ETL pipelines",
                "Work with big data technologies (Spark, Hadoop)",
                "Optimize database queries and performance"
            ],
            "software_engineer": [
                "Write clean, maintainable code",
                "Implement system design principles",
                "Collaborate effectively in development teams"
            ],
            "ml_engineer": [
                "Deploy ML models to production",
                "Optimize model performance and efficiency",
                "Implement MLOps best practices"
            ]
        }
        
        return base_metrics + role_specific_metrics.get(target_role.lower(), [])
    
    def get_career_transition_path(self, from_role: str, to_role: str) -> Dict[str, Any]:
        """
        Get a career transition path between two roles.
        """
        transition_skills = self.career_transitions.get(from_role.lower(), {}).get(to_role.lower(), [])
        
        if not transition_skills:
            # Fallback: get skills for target role
            transition_skills = self.get_target_skills_for_role(to_role)
        
        # Get learning path for transition skills
        learning_path = []
        for skill in transition_skills:
            # For single skill, we need to provide current skills and target skills
            # Since this is a career transition, assume basic skills as starting point
            basic_skills = ['programming fundamentals', 'data structures']
            path_result = self.skill_graph.get_skill_path(basic_skills, [skill])
            if path_result and 'learning_paths' in path_result:
                skill_path = path_result['learning_paths'].get(skill, {})
                if 'path' in skill_path:
                    learning_path.extend(skill_path['path'])
        
        # Remove duplicates
        unique_path = list(dict.fromkeys(learning_path))
        
        return {
            "from_role": from_role,
            "to_role": to_role,
            "transition_skills": transition_skills,
            "learning_path": unique_path,
            "stages": self._create_learning_stages(unique_path),
            "estimated_time": self._estimate_total_time(unique_path),
            "success_metrics": self._define_success_metrics(to_role)
        }
