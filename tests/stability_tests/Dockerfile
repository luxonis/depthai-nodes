# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

RUN apt install libgl1 -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Clone the depthai-nodes repository
RUN git clone https://github.com/luxonis/depthai-nodes.git

# Set the working directory to the cloned repository
WORKDIR /app/depthai-nodes

# Install the depthai-nodes package
RUN pip install -e .

# Install the development requirements
RUN pip install -r requirements-dev.txt

# Install pytest-xdist
RUN pip install pytest-xdist

# Set the working directory to the tests directory
WORKDIR /app/depthai-nodes/tests/stability_tests

# Run the setup script and the tests
ENTRYPOINT ["bash", "-c", "git checkout main && git pull && \
  echo 'Checking out branch' $BRANCH && \
  git checkout $BRANCH && \
  pip install --extra-index-url $LUXONIS_EXTRA_INDEX_URL depthai==$DEPTHAI_VERSION && \
  $MAIN_COMMAND"]