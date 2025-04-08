FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy application code and models
COPY . .

# Make port 8080 available
EXPOSE 8080

# Use gunicorn as the production server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app