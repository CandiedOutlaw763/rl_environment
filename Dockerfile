# Use an official, lightweight Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# ... (Keep the Python 3.12 and pip install steps the same) ...

# Expose the standard Hugging Face Space port
EXPOSE 7860

# Run the server application directly
CMD ["python", "server/app.py"]