# AI-Assisted Learning Recommender

A personalized learning path recommender system that helps tech professionals navigate career transitions and skill gaps using Gen AI.

## 🎯 Problem Solved

Professionals with career transitions and gaps struggle with personalized upskilling paths. This system creates custom learning plans by analyzing user profiles and recommending relevant resources.

## 🏗️ Architecture

- **Backend**: FastAPI with async processing
- **Frontend**: Streamlit for rapid UI development
- **Data Processing**: Pandas for ETL operations
- **AI/ML**: HuggingFace transformers for personalized recommendations
- **Database**: Snowflake for scalable data storage
- **Deployment**: Docker containerization

## 🚀 Features

- **Skill Gap Analysis**: Identify missing skills for target roles
- **Personalized Learning Paths**: AI-generated custom study plans
- **Resource Recommendations**: Curated learning materials and courses
- **Career Transition Support**: Specialized paths for DE→SDE, Frontend→Full Stack, etc.

## 📁 Project Structure

```
learning-recommender/
├── backend/          # FastAPI backend
│   └── main.py      # Main API endpoints
├── frontend/         # Streamlit frontend
│   └── app.py       # Main Streamlit application
├── data/            # Data processing scripts
│   ├── skills_processor.py  # Tech skills data processor
│   └── .gitkeep     # Directory placeholder
├── models/          # ML models and AI components
│   └── .gitkeep     # Directory placeholder
├── notebooks/       # Jupyter notebooks for exploration
├── tests/           # Test suite
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker configuration
├── docker-compose.yml # Docker compose setup
└── README.md        # Project documentation
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy, Scikit-learn
- **AI**: Transformers, PyTorch, Sentence-Transformers
- **Database**: Snowflake, SQLAlchemy
- **DevOps**: Docker, Git

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional)

### Local Development
1. **Clone the repository**
   ```bash
   git clone https://github.com/rajasreekatari/AI-assisted-learning-recommender.git
   cd AI-assisted-learning-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend**
   ```bash
   uvicorn backend.main:app --reload
   ```

4. **Run the frontend**
   ```bash
   streamlit run frontend/app.py
   ```

### Docker Deployment
1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 📊 Data Sources

- LinkedIn job postings (filtered for tech roles)
- GitHub Jobs API
- Stack Overflow job data
- Custom tech skill taxonomy

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /profile/create` - Create user profile
- `GET /profile/{profile_id}` - Get user profile
- `POST /recommend/path` - Generate learning path

### Data Endpoints
- `GET /skills/tech-taxonomy` - Get tech skills taxonomy
- `GET /career-paths` - Get career transition paths

## 🧪 Testing

Run the test suite:
```bash
pytest
```

## 📈 Data Processing

The system includes a data processor (`data/skills_processor.py`) that:
- Filters job data for tech professionals
- Extracts relevant skills using keyword matching
- Creates comprehensive skills analysis
- Supports multiple data sources (CSV files)

## 🤝 Contributing

This is a personal project showcasing skills in Data Engineering, Software Development, and Gen AI integration.

## 📝 License

MIT License

## 🔮 Roadmap

- [ ] Integrate HuggingFace models for AI recommendations
- [ ] Add Snowflake database integration
- [ ] Implement user authentication
- [ ] Add learning progress tracking
- [ ] Create mobile app version
- [ ] Add more career transition paths
- [ ] Implement skill gap visualization
- [ ] Add learning resource curation