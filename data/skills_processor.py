import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import json

class TechSkillsProcessor:
    """Process and filter skills data for tech professionals"""
    
    def __init__(self):
        # Tech-related keywords for filtering
        self.tech_keywords = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'c#', 'php', 'ruby',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'bash', 'powershell'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'rails',
                'asp.net', 'node.js', 'jquery', 'bootstrap', 'tailwind', 'material-ui'
            ],
            'databases': [
                'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'snowflake', 'bigquery', 'dynamodb', 'oracle', 'sql server', 'sqlite'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab',
                'github actions', 'circleci', 'travis ci', 'ansible', 'chef', 'puppet'
            ],
            'data_engineering': [
                'apache spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'pandas', 'numpy', 'scipy',
                'scikit-learn', 'tensorflow', 'pytorch', 'hive', 'impala', 'presto'
            ],
            'devops': [
                'ci/cd', 'continuous integration', 'continuous deployment', 'microservices',
                'rest api', 'graphql', 'soap', 'web services', 'load balancing', 'monitoring'
            ]
        }
        
        # Tech job titles
        self.tech_job_titles = [
            'software engineer', 'developer', 'programmer', 'data engineer', 'data scientist',
            'machine learning engineer', 'devops engineer', 'site reliability engineer',
            'frontend developer', 'backend developer', 'full stack developer', 'mobile developer',
            'ios developer', 'android developer', 'web developer', 'systems engineer',
            'cloud engineer', 'security engineer', 'qa engineer', 'test engineer',
            'product manager', 'technical lead', 'architect', 'cto', 'vp engineering'
        ]
    
    def load_data(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load data from multiple CSV files"""
        data = {}
        
        for name, path in file_paths.items():
            try:
                print(f"Loading {name} from {path}...")
                # Read in chunks for large files
                if path.endswith('job_summary.csv'):
                    # For very large files, read in chunks
                    chunk_list = []
                    chunk_size = 10000
                    
                    for chunk in pd.read_csv(path, chunksize=chunk_size):
                        chunk_list.append(chunk)
                    
                    data[name] = pd.concat(chunk_list, ignore_index=True)
                    print(f"Loaded {len(data[name])} rows from {name}")
                else:
                    data[name] = pd.read_csv(path)
                    print(f"Loaded {len(data[name])} rows from {name}")
                    
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
                data[name] = pd.DataFrame()
        
        return data
    
    def filter_tech_jobs(self, df: pd.DataFrame, title_column: str = 'job_title') -> pd.DataFrame:
        """Filter jobs to only include tech-related positions"""
        if title_column not in df.columns:
            print(f"Warning: Column '{title_column}' not found in dataframe")
            return df
        
        # Create a pattern to match tech job titles
        tech_pattern = '|'.join(self.tech_job_titles)
        
        # Filter rows where job title contains tech keywords
        tech_mask = df[title_column].str.lower().str.contains(tech_pattern, na=False)
        tech_jobs = df[tech_mask].copy()
        
        print(f"Found {len(tech_jobs)} tech jobs out of {len(df)} total jobs")
        return tech_jobs
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        extracted_skills = []
        
        # Check each category of skills
        for category, skills in self.tech_keywords.items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills.append(skill)
        
        return list(set(extracted_skills))  # Remove duplicates
    
    def process_job_skills(self, df: pd.DataFrame, skills_column: str = 'job_skills') -> pd.DataFrame:
        """Process and extract skills from job skills column"""
        if skills_column not in df.columns:
            print(f"Warning: Column '{skills_column}' not found in dataframe")
            return df
        
        print("Extracting skills from job descriptions...")
        
        # Extract skills from each job
        df['extracted_skills'] = df[skills_column].apply(self.extract_skills_from_text)
        df['skills_count'] = df['extracted_skills'].apply(len)
        
        # Filter jobs that have at least some tech skills
        df_with_skills = df[df['skills_count'] > 0].copy()
        
        print(f"Found {len(df_with_skills)} jobs with tech skills out of {len(df)} total jobs")
        return df_with_skills
    
    def create_skills_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """Create comprehensive skills analysis"""
        print("Creating skills analysis...")
        
        # Flatten all skills
        all_skills = []
        for skills_list in df['extracted_skills']:
            all_skills.extend(skills_list)
        
        # Count skill frequencies
        skill_counts = pd.Series(all_skills).value_counts()
        
        # Skills by category
        skills_by_category = {}
        for category, skills in self.tech_keywords.items():
            category_skills = {}
            for skill in skills:
                if skill in skill_counts.index:
                    category_skills[skill] = skill_counts[skill]
            skills_by_category[category] = category_skills
        
        # Top skills overall
        top_skills = skill_counts.head(50).to_dict()
        
        # Skills distribution
        skills_distribution = {
            'total_jobs': len(df),
            'jobs_with_skills': len(df[df['skills_count'] > 0]),
            'total_skills_found': len(all_skills),
            'unique_skills': len(skill_counts),
            'avg_skills_per_job': df['skills_count'].mean()
        }
        
        return {
            'skills_distribution': skills_distribution,
            'top_skills': top_skills,
            'skills_by_category': skills_by_category,
            'all_skills': skill_counts.to_dict()
        }
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to CSV"""
        print(f"Saving processed data to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
    
    def save_analysis(self, analysis: Dict[str, any], output_path: str):
        """Save skills analysis to JSON"""
        print(f"Saving analysis to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {output_path}")

def main():
    """Main function to process the skills data"""
    print("🚀 Starting Tech Skills Data Processing...")
    
    # Initialize processor
    processor = TechSkillsProcessor()
    
    # Define file paths (update these to match your actual file locations)
    file_paths = {
        'job_skills': r'C:\Users\katar\Downloads\skills data\job_skills.csv',
        'linkedin_jobs': r'C:\Users\katar\Downloads\skills data\linkedin_job_postings.csv',
        'job_summary': r'C:\Users\katar\Downloads\skills data\job_summary.csv'
    }
    
    try:
        # Load data
        print("📊 Loading datasets...")
        data = processor.load_data(file_paths)
        
        # Process each dataset
        processed_data = {}
        
        for name, df in data.items():
            if df.empty:
                continue
                
            print(f"\n🔍 Processing {name}...")
            
            # Handle different dataset types
            if name == 'job_skills':
                # This dataset has skills but no job titles
                # We'll need to join it with linkedin_jobs to get job titles
                print("Processing job_skills dataset (skills only)")
                if 'job_skills' in df.columns:
                    processed_df = processor.process_job_skills(df, 'job_skills')
                    processed_data[name] = processed_df
                    
                    # Save processed data
                    output_file = f"data/processed_{name}.csv"
                    processor.save_processed_data(processed_df, output_file)
                else:
                    print(f"No skills column found in {name}")
                    
            elif name == 'linkedin_jobs':
                # This dataset has job titles but no skills
                print("Processing linkedin_jobs dataset (job titles only)")
                if 'job_title' in df.columns:
                    tech_jobs = processor.filter_tech_jobs(df, 'job_title')
                    processed_data[name] = tech_jobs
                    
                    # Save processed data
                    output_file = f"data/processed_{name}.csv"
                    processor.save_processed_data(tech_jobs, output_file)
                else:
                    print(f"No job title column found in {name}")
                    
            elif name == 'job_summary':
                # This dataset has job descriptions
                print("Processing job_summary dataset (job descriptions)")
                if 'job_summary' in df.columns:
                    # Extract skills from job summaries
                    df['extracted_skills'] = df['job_summary'].apply(processor.extract_skills_from_text)
                    df['skills_count'] = df['extracted_skills'].apply(len)
                    
                    # Filter jobs with tech skills
                    df_with_skills = df[df['skills_count'] > 0].copy()
                    processed_data[name] = df_with_skills
                    
                    # Save processed data
                    output_file = f"data/processed_{name}.csv"
                    processor.save_processed_data(df_with_skills, output_file)
                else:
                    print(f"No job summary column found in {name}")
        
        # Create combined analysis
        print("\n📈 Creating combined skills analysis...")
        if processed_data:
            # Combine all processed data
            combined_df = pd.concat(processed_data.values(), ignore_index=True)
            
            # Ensure we have the required columns for analysis
            if 'extracted_skills' not in combined_df.columns:
                print("Creating extracted_skills column from available data...")
                if 'job_skills' in combined_df.columns:
                    combined_df['extracted_skills'] = combined_df['job_skills'].apply(processor.extract_skills_from_text)
                elif 'job_summary' in combined_df.columns:
                    combined_df['extracted_skills'] = combined_df['job_summary'].apply(processor.extract_skills_from_text)
                else:
                    print("No skills data available for analysis")
                    return
                
                combined_df['skills_count'] = combined_df['extracted_skills'].apply(len)
            
            combined_analysis = processor.create_skills_analysis(combined_df)
            
            # Save analysis
            processor.save_analysis(combined_analysis, "data/tech_skills_analysis.json")
            
            print(f"\n✅ Processing complete!")
            print(f"📊 Total tech jobs processed: {len(combined_df)}")
            print(f"🛠️ Total skills extracted: {combined_analysis['skills_distribution']['total_skills_found']}")
            print(f"📁 Output files saved in 'data/' directory")
            
            # Show some sample data
            print(f"\n📋 Sample of processed data:")
            print(f"Columns: {combined_df.columns.tolist()}")
            print(f"Sample skills: {combined_df['extracted_skills'].head(3).tolist()}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
