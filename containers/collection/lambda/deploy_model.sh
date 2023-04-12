#!/bin/bash
set -e

if [ -z "${AWS_LAMBDA_CONTAINER_REPO}" ]; then
    echo "ERROR: AWS_LAMBDA_CONTAINER_REPO variable not set."
    exit 1
fi

if [ -z "${AWS_LAMBDA_REGION}" ]; then
    echo "ERROR: AWS_LAMBDA_REGION variable not set."
fi

# remove anything unrelated from the staging directory
rm -r -f containers/collection/lambda/staged_models

mkdir containers/collection/lambda/staged_models

# copy the trained model over
cp -r trained_models/$1 containers/collection/lambda/staged_models

# get rid of any old images so we don't wind up with dangling ones
docker image rm collection:$1 -f
docker image rm $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection:$1 -f

# build the image
docker build -t collection:$1 -f containers/collection/lambda/Dockerfile-model .

# tag the image for aws, get login details, and then push it up
# this requires AWS CLI to be setup locally
docker tag collection:$1 $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection:$1
aws ecr get-login-password --region $AWS_LAMBDA_REGION | docker login --username AWS --password-stdin $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com
docker push $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection:$1

# cleanup the staging directory
rm -r -f containers/collection/lambda/staged_models
