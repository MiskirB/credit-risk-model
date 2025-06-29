# Use a lightweight Python image
# Changed from 3.9-slim to 3.13-slim to match the Python version the MLflow model was saved with (Python 3.13.2)
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
# IMPORTANT: Ensure your requirements.txt lists the EXACT versions of dependencies
# that the MLflow model requires (e.g., numpy==2.3.1, psutil==7.0.0, scikit-learn==1.7.0, scipy==1.16.0)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY src/ ./src/
COPY model/ ./model/

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.main_render:app", "--host", "0.0.0.0", "--port", "8000"]
