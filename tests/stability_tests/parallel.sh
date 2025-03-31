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

# Delay between iterations in seconds
LOOP_DELAY=1

# Extract all model names from the config.py file into an array
models=()
while IFS= read -r line; do
    models+=("$line")
done < <(grep -o '"luxonis/[^"]*"' config.py | tr -d '"')

# Print the full list as a single line with commas
# echo "All models: [$(IFS=,; echo "${models[*]}")]"
echo "Total models: ${#models[@]}"

# counter
counter=0

# Iterate over each model
for model in "${models[@]}"; do
  echo "$model"
  echo ""
  counter=$((counter + 1))

  command_to_run=""

  if [ $counter -eq 1 ]; then
    command_to_run="hil --testbed test2 --skip-sanity-check --stability-test --stability-name=depthai-nodes-parallel-$counter --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"echo $GHCR_TOKEN | docker login ghcr.io -u $GHCR_USERNAME --password-stdin\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env FLAGS=\"-m $model --duration $TEST_DURATION\"'"
  else
    command_to_run="hil --testbed test2 --skip-sanity-check --hold-reservation --stability-test --stability-name=depthai-nodes-parallel-$counter --wait --reservation-name $RESERVATION_NAME --before-docker-pull \"echo $GHCR_TOKEN | docker login ghcr.io -u $GHCR_USERNAME --password-stdin\" --docker-image ghcr.io/luxonis/depthai-nodes-stability-tests --docker-run-args '--env LUXONIS_EXTRA_INDEX_URL=$LUXONIS_EXTRA_INDEX_URL --env DEPTHAI_VERSION=$DEPTHAI_VERSION --env B2_APPLICATION_KEY=$B2_APPLICATION_KEY --env B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID --env BRANCH=$BRANCH --env FLAGS=\"-m $model --duration $TEST_DURATION\"'"
  fi

  echo "$command_to_run"

  eval "$command_to_run"

  echo ""

  # Add delay between iterations if not the last model
  if [ $counter -lt ${#models[@]} ]; then
    echo "Waiting for $LOOP_DELAY seconds before starting the next test..."
    sleep $LOOP_DELAY
  fi
done