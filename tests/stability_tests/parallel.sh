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

# Define all available testbeds
testbeds=("test" "test2")
# testbeds=("test" "test2" "slo4012-4s" "slo407-4d") # TODO: Uncomment this when slo4012-4s and slo407-4d are available again

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
done < <(find ../unittests/test_nodes/test_host_nodes -type f -name 'test_*_node.py')

# Print the total number of test files
echo "Total test files for HostNodes: ${#test_host_node_files[@]}"

# Extract all test files from the test_nodes/test_threaded_host_nodes directory
test_threaded_host_node_files=()

# List Python files in ../unittests/test_nodes/test_threaded_host_nodes that start with test_ and store them in an array
while IFS= read -r file; do
    test_threaded_host_node_files+=("$file")
done < <(find ../unittests/test_nodes/test_threaded_host_nodes -type f -name 'test_*_node.py')

# Print the total number of test files
echo "Total test files for ThreadedHostNodes: ${#test_threaded_host_node_files[@]}"

# Combine all tests into a single array
all_tests=()

# Add models with a prefix to identify them
for model in "${models[@]}"; do
    all_tests+=("model:$model")
done

# Add host node tests with a prefix
for test_file in "${test_host_node_files[@]}"; do
    all_tests+=("host:$test_file")
done

# Add threaded host node tests with a prefix
for test_file in "${test_threaded_host_node_files[@]}"; do
    all_tests+=("threaded:$test_file")
done

echo "Total tests to run: ${#all_tests[@]}"

# Calculate tests per testbed (rounded up)
total_tests=${#all_tests[@]}
testbed_count=${#testbeds[@]}
tests_per_testbed=$(( (total_tests + testbed_count - 1) / testbed_count ))

echo "Distributing approximately $tests_per_testbed tests per testbed"

# Function to run a test based on its type
run_test() {
    local test=$1
    local testbed=$2
    
    # Extract test type and content
    local test_type=${test%%:*}
    local test_content=${test#*:}
    
    echo "--------------------------------"
    
    if [[ "$test_type" == "model" ]]; then
        # Handle model test
        model_name="${test_content%%:*}"
        model_name="${model_name##*/}"
        
        echo "Running $model_name test on testbed $testbed"
        echo "--------------------------------"
        
        command_to_run="hil --testbed $testbed --skip-sanity-check --stability-test --stability-name=depthai-nodes-$model_name --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"DOCKER_CONFIG=$DOCKER_CONFIG_DIR\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--memory=256m --env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env MAIN_COMMAND=\"python main.py -m $test_content --duration $TEST_DURATION\"'"
    else
        # Handle host node or threaded host node test
        test_file_name=$(basename "$test_content" .py)
        
        echo "Running $test_file_name test on testbed $testbed"
        echo "--------------------------------"
        
        command_to_run="hil --testbed $testbed --skip-sanity-check --stability-test --stability-name=depthai-nodes-$test_file_name --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"DOCKER_CONFIG=$DOCKER_CONFIG_DIR\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--memory=256m --env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env MAIN_COMMAND=\"pytest $test_content --duration $TEST_DURATION -n 3 -r a --log-cli-level=DEBUG --color=yes -s\"'"
    fi
    
    echo "$command_to_run"
    eval "$command_to_run"
    
    echo "--------------------------------"
}

# Run tests in parallel across testbeds
for ((i=0; i<${#all_tests[@]}; i++)); do
    # NOTE: Currently, we will only trigger 6 tests until HIL support is fixed
    if [ $i -gt 5 ]; then
        break
    fi
    testbed_index=$((i % testbed_count))
    testbed=${testbeds[$testbed_index]}
    
    run_test "${all_tests[$i]}" "$testbed"
    
    if [ $i -lt ${#all_tests[@]} ]; then
        echo "Waiting for $LOOP_DELAY seconds before running next test..."
        sleep $LOOP_DELAY
    fi
done

echo "All tests started. Check out the provided links to see the results."

# Clean up the temporary Docker config directory
rm -rf $DOCKER_CONFIG_DIR