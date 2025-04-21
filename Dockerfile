# Use the official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Streamlit app
COPY . .

# Expose Streamlit’s default port
EXPOSE 8502

# Launch the app
ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
