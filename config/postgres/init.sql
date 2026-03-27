-- ML Platform - PostgreSQL Initialization
-- Creates databases and users for MLflow, Airflow, and the platform.

-- MLflow backend database
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Airflow backend database
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Platform database (ML pipeline data, feature store, etc.)
CREATE DATABASE platform_db;
CREATE USER platform_user WITH PASSWORD 'platform_password';
GRANT ALL PRIVILEGES ON DATABASE platform_db TO platform_user;

-- Connect to mlflow_db and grant schema permissions
\connect mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;

-- Connect to airflow_db and grant schema permissions
\connect airflow_db;
GRANT ALL ON SCHEMA public TO airflow_user;

-- Connect to platform_db and grant schema permissions
\connect platform_db;
GRANT ALL ON SCHEMA public TO platform_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO platform_user;
