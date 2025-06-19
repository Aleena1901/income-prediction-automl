# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install system dependencies needed for some Python packages (e.g., libgomp1 for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run the Streamlit app when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 