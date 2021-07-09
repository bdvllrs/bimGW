#!/bin/bash
# Starts submitJob.sh from here: https://gist.github.com/bdvllrs/e59f276a563c3c9d790f33b27a8e4c9e

jobName=$1
scriptName=$2

# Copy dataset to tmpdir

if [[ ! -e "${DATASET_FOLDER}/imagenet" ]]; then
  if [[ -e "${WORK_DATASET_FOLDER}/imagenet" ]]; then
    echo "Copying dataset"
    cp -r "${WORK_DATASET_FOLDER}/imagenet" "${DATASET_FOLDER}/imagenet"
  fi
fi

bash ~/submitJob.sh "${BIM_ENV}" "${jobName}" "${BIM_WORKDIR}" "${RUN_WORKDIR}/BIM" "bimGW/scripts/${scriptName}.py" "${@:3}"