#!/bin/bash

GHCR_TOKEN=${GHCR_TOKEN:-""}
GHCR_USERNAME=${GHCR_USERNAME:-""}
RESERVATION_NAME=${RESERVATION_NAME:-"depthai-nodes-stability-tests"}
LUXONIS_EXTRA_INDEX_URL=${LUXONIS_EXTRA_INDEX_URL:-""}
DEPTHAI_VERSION=${DEPTHAI_VERSION:-""}
B2_APPLICATION_KEY=${B2_APPLICATION_KEY:-""}
B2_APPLICATION_KEY_ID=${B2_APPLICATION_KEY_ID:-""}
BRANCH=${BRANCH:-"main"}
TEST_DURATION=${TEST_DURATION:-"10"}

testbed="test2"

# Delay between iterations in seconds
LOOP_DELAY=2

# Create a temporary Docker config directory
DOCKER_CONFIG_DIR=$(mktemp -d)
echo "{\"auths\":{\"ghcr.io\":{\"auth\":\"$(echo -n "$DOCKER_USERNAME:$DOCKER_PASSWORD" | base64)\"}}}" > $DOCKER_CONFIG_DIR/config.json


# Extract all model names from the config.py file into an array
models=()
while IFS= read -r line; do
    models+=("$line")
done < <(grep -o '"luxonis/[^"]*"' config.py | tr -d '"')

# Print the full list as a single line with commas
# echo "All models: [$(IFS=,; echo "${models[*]}")]"
echo "Total models: ${#models[@]}"

# Extract all test files from the test_host_nodes directory
test_host_node_files=()

# List Python files in ../unittests/test_nodes/test_host_nodes that start with test_ and store them in an array
while IFS= read -r file; do
    test_host_node_files+=("$file")
done < <(find ../unittests/test_nodes/test_host_nodes -type f -name 'test_*.py')

# Print the total number of test files
echo "Total test files for HostNodes: ${#test_host_node_files[@]}"

# Extract all test files from the test_nodes/test_threaded_host_nodes directory
test_threaded_host_node_files=()

# List Python files in ../unittests/test_nodes/test_threaded_host_nodes that start with test_ and store them in an array
while IFS= read -r file; do
    test_threaded_host_node_files+=("$file")
done < <(find ../unittests/test_nodes/test_threaded_host_nodes -type f -name 'test_*.py')

# Print the total number of test files
echo "Total test files for ThreadedHostNodes: ${#test_threaded_host_node_files[@]}"

# counter
counter=0

# Iterate over each model
for model in "${models[@]}"; do
  echo "$model"
  echo ""
  counter=$((counter + 1))

  command_to_run="hil --testbed $testbed --skip-sanity-check --stability-test --stability-name=depthai-nodes-parallel-$counter --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"DOCKER_CONFIG=$DOCKER_CONFIG_DIR\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env MAIN_COMMAND=\"python main.py -m $model --duration $TEST_DURATION\"'"

  echo "$command_to_run"

  eval "$command_to_run"

  echo ""

  # Add delay between iterations if not the last model
  if [ $counter -lt ${#models[@]} ]; then
    echo "Waiting for $LOOP_DELAY seconds before starting the next test..."
    sleep $LOOP_DELAY
  fi
done

# Iterate over each test file for HostNodes
for test_file in "${test_host_node_files[@]}"; do
  echo "Running test file: $test_file"
  counter=$((counter + 1))
  command_to_run="hil --testbed $testbed --skip-sanity-check --stability-test --stability-name=depthai-nodes-parallel-$counter --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"DOCKER_CONFIG=$DOCKER_CONFIG_DIR\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env MAIN_COMMAND=\"pytest $test_file --duration $TEST_DURATION -n 3 -r a --log-cli-level=DEBUG --color=yes -s\"'"
  
  eval "$command_to_run"

  echo ""

  # Add delay between iterations if not the last model
  if [ $counter -lt ${#models[@]} ]; then
    echo "Waiting for $LOOP_DELAY seconds before starting the next test..."
    sleep $LOOP_DELAY
  fi
done

# Iterate over each test file for ThreadedHostNodes
for test_file in "${test_threaded_host_node_files[@]}"; do
  echo "Running test file: $test_file"
  counter=$((counter + 1))
  command_to_run="hil --testbed $testbed --skip-sanity-check --stability-test --stability-name=depthai-nodes-parallel-$counter --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"DOCKER_CONFIG=$DOCKER_CONFIG_DIR\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env MAIN_COMMAND=\"pytest $test_file --duration $TEST_DURATION -n 3 -r a --log-cli-level=DEBUG --color=yes -s\"'"
  
  eval "$command_to_run"

  echo ""

  # Add delay between iterations if not the last model
  if [ $counter -lt ${#models[@]} ]; then
    echo "Waiting for $LOOP_DELAY seconds before starting the next test..."
    sleep $LOOP_DELAY
  fi
done

echo "All tests started. Check out the provided links to see the results."

# Clean up the temporary Docker config directory
rm -rf $DOCKER_CONFIG_DIR