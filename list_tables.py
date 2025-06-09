from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Validate configuration
missing_vars = [var for var, value in [
    ("DB_USERNAME", DB_USERNAME),
    ("DB_PASSWORD", DB_PASSWORD),
    ("DB_HOST", DB_HOST),
    ("DB_PORT", DB_PORT),
    ("DB_NAME", DB_NAME)
] if not value]
if missing_vars:
    logger.error(f"Missing database configuration: {', '.join(missing_vars)}")
    raise ValueError(f"Database configuration incomplete: {missing_vars}")

# Database URL
DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

def list_tables():
    """List all tables in the public schema."""
    try:
        engine = create_engine(DATABASE_URL, connect_args={"options": "-c default_transaction_read_only=true"})
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            logger.info(f"Listed tables: {tables}")
            return tables
    except SQLAlchemyError as e:
        logger.error(f"Error listing tables: {str(e)}")
        return []

if __name__ == "__main__":
    tables = list_tables()
    print(tables)