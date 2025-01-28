# Use the official Python 3.8 image from the Docker Hub
ARG PYTHON_VERSION
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Clone the depthai-nodes repository
RUN git clone https://github.com/luxonis/depthai-nodes.git

# Set the working directory to the cloned repository
WORKDIR /app/depthai-nodes

# Install the depthai-nodes package
RUN pip install -e .

# Install the development requirements
RUN pip install -r requirements-dev.txt

# Set the working directory to the tests directory
WORKDIR /app/depthai-nodes/tests/integration_tests

# Run the setup script and the tests

# CMD ["bash"]
ENTRYPOINT ["bash", "-c", "git checkout main && git pull && git checkout $BRANCH && git pull && pip install --extra-index-url $LUXONIS_EXTRA_INDEX_URL depthai==$DEPTHAI_VERSION && python main.py $FLAGS"]
# ENTRYPOINT ["bash", "-c", "git checkout main && git pull && git checkout $BRANCH && git pull && pip install --extra-index-url $LUXONIS_EXTRA_INDEX_URL depthai==$DEPTHAI_VERSION &&  source <(python setup_camera_ips.py) && python main.py -all"]