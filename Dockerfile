# Use an official, lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
# (This includes your /data folder with the CSVs, so it won't crash!)
COPY . .

# Run the inference script when the container launches
CMD ["python", "inference.py"]