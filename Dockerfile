# Use Python 3.9 slim image as base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app


# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the application
CMD ["python", "-m", "streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
