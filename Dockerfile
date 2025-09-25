FROM nvcr.io/nvidia/tensorrt:24.07-py3

LABEL maintainer="Aditri Singh <aditwisingh@gmail.com>"
LABEL version="1.0"
LABEL description="Real-time UAV detection using TensorRT and OpenCV"

WORKDIR /app

# Install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Default command to start bash (users can run tracker manually)
CMD ["bash"]
