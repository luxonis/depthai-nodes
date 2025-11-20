# Use the official Python 3.8 image from the Docker Hub
FROM python:3.12-slim

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
WORKDIR /app/depthai-nodes/tests/end_to_end

# Run the setup script and the tests
ENTRYPOINT ["bash", "-c", "git checkout main && git pull && \
  if git show-ref --tags --quiet $BRANCH; then \
    echo 'Checking out tag' $BRANCH && \
    git checkout tags/$BRANCH; \
  else \
    echo 'Checking out branch' $BRANCH && \
    git checkout $BRANCH; \
  fi && \
  pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ --extra-index-url $LUXONIS_EXTRA_INDEX_URL  depthai==$DEPTHAI_VERSION && \
  source <(python setup_camera_ips.py) && \
  DEPTHAI_NODES_LEVEL=debug python -u main.py $FLAGS"]