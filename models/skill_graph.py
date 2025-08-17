import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import json

class SkillDependencyGraph:
    """
    DSA Component: Skill dependency graph using NetworkX
    Implements graph algorithms for skill path recommendations
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.skill_levels = {
            'beginner': ['python', 'sql', 'html', 'css', 'javascript', 'git'],
            'intermediate': ['django', 'flask', 'react', 'node.js', 'postgresql', 'mongodb'],
            'advanced': ['kubernetes', 'docker', 'aws', 'apache spark', 'airflow', 'system design'],
            'expert': ['distributed systems', 'machine learning', 'ai architecture', 'scalability']
        }
        
        # Initialize skill dependencies
        self._build_skill_dependencies()
    
    def _build_skill_dependencies(self):
        """Build the skill dependency graph with edges representing prerequisites"""
        
        # Programming Language Dependencies
        language_deps = {
            'python': ['programming fundamentals'],
            'java': ['programming fundamentals'],
            'javascript': ['html', 'css'],
            'typescript': ['javascript'],
            'react': ['javascript', 'html', 'css'],
            'angular': ['typescript', 'javascript'],
            'vue': ['javascript', 'html', 'css'],
            'node.js': ['javascript'],
            'django': ['python'],
            'flask': ['python'],
            'spring': ['java'],
            'express': ['javascript', 'node.js'],
            'laravel': ['php'],
            'rails': ['ruby']
        }
        
        # Database Dependencies
        db_deps = {
            'postgresql': ['sql'],
            'mysql': ['sql'],
            'mongodb': ['json', 'javascript'],
            'redis': ['data structures'],
            'elasticsearch': ['json', 'search algorithms'],
            'snowflake': ['sql', 'data warehousing'],
            'bigquery': ['sql', 'data warehousing']
        }
        
        # Cloud & DevOps Dependencies
        cloud_deps = {
            'docker': ['linux', 'networking'],
            'kubernetes': ['docker', 'networking'],
            'aws': ['networking', 'linux'],
            'azure': ['networking', 'linux'],
            'gcp': ['networking', 'linux'],
            'terraform': ['yaml', 'networking'],
            'jenkins': ['java', 'scripting'],
            'gitlab': ['git'],
            'github actions': ['git', 'yaml']
        }
        
        # Data Engineering Dependencies
        de_deps = {
            'apache spark': ['python', 'java', 'distributed systems'],
            'hadoop': ['java', 'distributed systems'],
            'kafka': ['java', 'distributed systems'],
            'airflow': ['python', 'workflow management'],
            'dbt': ['sql', 'data modeling'],
            'pandas': ['python'],
            'numpy': ['python'],
            'scikit-learn': ['python', 'pandas', 'numpy', 'mathematics']
        }
        
        # Machine Learning Dependencies
        ml_deps = {
            'tensorflow': ['python', 'mathematics', 'linear algebra'],
            'pytorch': ['python', 'mathematics', 'linear algebra'],
            'hugging face': ['python', 'transformers', 'nlp'],
            'mlflow': ['python', 'mlops'],
            'scikit-learn': ['python', 'pandas', 'numpy', 'mathematics']
        }
        
        # Combine all dependencies
        all_deps = {**language_deps, **db_deps, **cloud_deps, **de_deps, **ml_deps}
        
        # Add nodes and edges to graph
        for skill, prerequisites in all_deps.items():
            self.graph.add_node(skill, skill_type='technical')
            
            for prereq in prerequisites:
                self.graph.add_node(prereq, skill_type='foundational')
                self.graph.add_edge(prereq, skill, relationship='prerequisite')
        
        # Add foundational skills
        foundational_skills = [
            'programming fundamentals', 'data structures', 'algorithms',
            'networking', 'linux', 'git', 'sql', 'html', 'css',
            'mathematics', 'linear algebra', 'workflow management',
            'data modeling', 'data warehousing', 'distributed systems',
            'scripting', 'yaml', 'json', 'search algorithms', 'mlops'
        ]
        
        for skill in foundational_skills:
            if skill not in self.graph:
                self.graph.add_node(skill, skill_type='foundational')
    
    def get_skill_path(self, current_skills: List[str], target_skills: List[str]) -> Dict:
        """
        Find the shortest learning path from current skills to target skills
        Uses BFS algorithm as specified in the original plan
        """
        if not current_skills or not target_skills:
            return {"error": "Both current and target skills are required"}
        
        # Find missing skills
        missing_skills = [skill for skill in target_skills if skill not in current_skills]
        
        if not missing_skills:
            return {"message": "You already have all target skills!", "path": []}
        
        # Build subgraph for path finding
        all_relevant_skills = set(current_skills + target_skills)
        
        # Find all skills that are prerequisites for target skills
        for target in target_skills:
            if target in self.graph:
                # Get all ancestors (prerequisites)
                ancestors = nx.ancestors(self.graph, target)
                all_relevant_skills.update(ancestors)
        
        # Create subgraph for path finding
        subgraph = self.graph.subgraph(all_relevant_skills)
        
        # Find shortest paths using BFS
        learning_paths = {}
        for target in missing_skills:
            if target in subgraph:
                try:
                    # Find shortest path from any current skill to target
                    shortest_path = None
                    min_length = float('inf')
                    
                    for current_skill in current_skills:
                        if current_skill in subgraph and nx.has_path(subgraph, current_skill, target):
                            path = nx.shortest_path(subgraph, current_skill, target)
                            if len(path) < min_length:
                                min_length = len(path)
                                shortest_path = path
                    
                    if shortest_path:
                        learning_paths[target] = {
                            'path': shortest_path[1:],  # Exclude starting skill
                            'length': len(shortest_path) - 1,
                            'prerequisites': list(nx.ancestors(subgraph, target))
                        }
                except nx.NetworkXNoPath:
                    learning_paths[target] = {"error": f"No path found to {target}"}
        
        return {
            "missing_skills": missing_skills,
            "learning_paths": learning_paths,
            "total_skills_to_learn": len(missing_skills),
            "estimated_time": self._estimate_learning_time(learning_paths)
        }
    
    def _estimate_learning_time(self, learning_paths: Dict) -> str:
        """Estimate learning time based on skill complexity and dependencies"""
        total_weeks = 0
        
        for skill, path_info in learning_paths.items():
            if 'length' in path_info:
                # Base time: 2 weeks per skill level
                skill_levels = path_info['length']
                total_weeks += skill_levels * 2
        
        if total_weeks <= 8:
            return f"{total_weeks} weeks"
        elif total_weeks <= 24:
            months = total_weeks // 4
            return f"{months} months"
        else:
            years = total_weeks // 52
            return f"{years} year(s)"
    
    def get_skill_dependencies(self, skill: str) -> Dict:
        """Get all dependencies for a specific skill"""
        if skill not in self.graph:
            return {"error": f"Skill '{skill}' not found in graph"}
        
        dependencies = {
            "skill": skill,
            "prerequisites": list(nx.ancestors(self.graph, skill)),
            "dependents": list(nx.descendants(self.graph, skill)),
            "skill_type": self.graph.nodes[skill].get('skill_type', 'unknown'),
            "in_degree": self.graph.in_degree(skill),
            "out_degree": self.graph.out_degree(skill)
        }
        
        return dependencies
    
    def visualize_skill_graph(self, skills: List[str] = None, output_path: str = None):
        """Visualize the skill dependency graph"""
        if skills:
            # Create subgraph for specific skills
            subgraph = self.graph.subgraph(skills)
            G = subgraph
        else:
            G = self.graph
        
        # Create plotly visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color nodes by skill type
            skill_type = G.nodes[node].get('skill_type', 'unknown')
            if skill_type == 'foundational':
                node_colors.append('#1f77b4')  # Blue
            elif skill_type == 'technical':
                node_colors.append('#ff7f0e')  # Orange
            else:
                node_colors.append('#2ca02c')  # Green
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            ))
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Skill Dependency Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def get_career_transition_path(self, from_role: str, to_role: str) -> Dict:
        """Get learning path for career transition"""
        
        # Define role skill mappings
        role_skills = {
            'data_analyst': ['python', 'sql', 'pandas', 'numpy', 'excel'],
            'data_engineer': ['python', 'sql', 'apache spark', 'airflow', 'kafka', 'hadoop'],
            'software_engineer': ['python', 'java', 'javascript', 'data structures', 'algorithms', 'system design'],
            'frontend_developer': ['html', 'css', 'javascript', 'react', 'angular', 'vue'],
            'backend_developer': ['python', 'java', 'node.js', 'databases', 'apis'],
            'fullstack_developer': ['html', 'css', 'javascript', 'python', 'java', 'databases', 'apis'],
            'devops_engineer': ['linux', 'docker', 'kubernetes', 'aws', 'jenkins', 'terraform'],
            'machine_learning_engineer': ['python', 'mathematics', 'scikit-learn', 'tensorflow', 'pytorch']
        }
        
        if from_role not in role_skills or to_role not in role_skills:
            return {"error": "Role not found in skill mappings"}
        
        current_skills = role_skills[from_role]
        target_skills = role_skills[to_role]
        
        return self.get_skill_path(current_skills, target_skills)
    
    def get_advanced_career_transitions(self) -> Dict:
        """Get advanced career transition matrix with difficulty levels"""
        
        transitions = {
            'data_analyst': {
                'data_engineer': {'difficulty': 'medium', 'estimated_time': '6-12 months', 'key_skills': ['apache spark', 'airflow', 'kafka']},
                'data_scientist': {'difficulty': 'medium', 'estimated_time': '8-14 months', 'key_skills': ['statistics', 'machine_learning', 'deep_learning']},
                'business_intelligence': {'difficulty': 'easy', 'estimated_time': '3-6 months', 'key_skills': ['power_bi', 'tableau', 'data_modeling']}
            },
            'frontend_developer': {
                'fullstack_developer': {'difficulty': 'medium', 'estimated_time': '6-10 months', 'key_skills': ['python', 'databases', 'api_design']},
                'ui_ux_designer': {'difficulty': 'easy', 'estimated_time': '4-8 months', 'key_skills': ['design_principles', 'user_research', 'prototyping']},
                'mobile_developer': {'difficulty': 'medium', 'estimated_time': '6-12 months', 'key_skills': ['react_native', 'swift', 'kotlin']}
            },
            'software_engineer': {
                'data_engineer': {'difficulty': 'medium', 'estimated_time': '6-10 months', 'key_skills': ['sql', 'data_warehousing', 'etl']},
                'ml_engineer': {'difficulty': 'hard', 'estimated_time': '12-18 months', 'key_skills': ['mathematics', 'machine_learning', 'mlops']},
                'devops_engineer': {'difficulty': 'medium', 'estimated_time': '6-12 months', 'key_skills': ['docker', 'kubernetes', 'ci_cd']}
            }
        }
        
        return transitions
    
    def get_transition_recommendations(self, current_role: str, experience_years: int) -> Dict:
        """Get personalized transition recommendations based on experience"""
        
        if experience_years < 1:
            return {"message": "Focus on mastering current role fundamentals first", "recommended_transitions": []}
        elif experience_years < 3:
            return {
                "message": "Good time to explore adjacent roles",
                "recommended_transitions": ["data_analyst", "frontend_developer", "backend_developer"]
            }
        else:
            return {
                "message": "Ready for more complex transitions",
                "recommended_transitions": ["data_engineer", "ml_engineer", "devops_engineer", "fullstack_developer"]
            }
    
    def save_graph(self, filepath: str):
        """Save the graph to a file"""
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_graph(self, filepath: str):
        """Load the graph from a file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)

# Example usage and testing
if __name__ == "__main__":
    # Create skill graph
    skill_graph = SkillDependencyGraph()
    
    # Test skill path finding
    current_skills = ['python', 'sql']
    target_skills = ['django', 'postgresql']
    
    result = skill_graph.get_skill_path(current_skills, target_skills)
    print("Skill Path Result:", json.dumps(result, indent=2))
    
    # Test career transition
    career_path = skill_graph.get_career_transition_path('data_analyst', 'data_engineer')
    print("Career Transition Path:", json.dumps(career_path, indent=2))
    
    # Save graph
    skill_graph.save_graph('models/skill_dependency_graph.json')

