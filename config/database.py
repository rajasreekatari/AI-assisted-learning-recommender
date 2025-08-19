import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import snowflake.connector

class DatabaseConfig:
    """Database configuration for Snowflake connection"""
    
    def __init__(self):
        # Snowflake connection parameters
        self.account = "MOGMPHI-DZ49162"
        self.user = "RAJASREEK"
        self.password = os.getenv("SNOWFLAKE_PASSWORD")  # Set this as environment variable
        self.warehouse = "COMPUTE_WH"  # Default warehouse
        self.database = "SKILLS"
        self.schema = "PUBLIC"  # Default schema
        self.role = "ACCOUNTADMIN"
        
    def get_connection_string(self) -> str:
        """Generate SQLAlchemy connection string for Snowflake"""
        return (
            f"snowflake://{self.user}:{self.password}@"
            f"{self.account}/{self.database}/{self.schema}"
            f"?warehouse={self.warehouse}&role={self.role}"
        )
    
    def get_snowflake_connector(self):
        """Get direct Snowflake connector for raw SQL operations"""
        return snowflake.connector.connect(
            account=self.account,
            user=self.user,
            password=self.password,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
            role=self.role
        )

# Global database configuration instance
db_config = DatabaseConfig()

def get_database_engine():
    """Get SQLAlchemy engine for database operations"""
    try:
        engine = create_engine(
            db_config.get_connection_string(),
            echo=False,  # Set to True for SQL query logging
            pool_pre_ping=True,
            pool_recycle=3600
        )
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None

def get_database_session() -> Optional[Session]:
    """Get database session for ORM operations"""
    engine = get_database_engine()
    if engine:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    return None

def test_connection():
    """Test Snowflake connection"""
    try:
        conn = db_config.get_snowflake_connector()
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        version = cursor.fetchone()[0]
        print(f"✅ Snowflake connection successful! Version: {version}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Snowflake connection failed: {e}")
        return False
