# Dockerfile
FROM bitnami/spark:latest

# Install required Python packages
RUN pip install pandas