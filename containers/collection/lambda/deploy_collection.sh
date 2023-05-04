#!/bin/bash
set -e

if [ -z "${AWS_LAMBDA_CONTAINER_REPO}" ]; then
    echo "ERROR: AWS_LAMBDA_CONTAINER_REPO variable not set."
    exit 1
fi

if [ -z "${AWS_LAMBDA_REGION}" ]; then
    echo "ERROR: AWS_LAMBDA_REGION variable not set."
fi

# build the image
docker buildx build --platform linux/amd64 -f containers/collection/lambda/Dockerfile-base -t collection .

# tag the image
docker tag collection $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection
# get login details
aws ecr get-login-password --region $AWS_LAMBDA_REGION | docker login --username AWS --password-stdin $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection

# push the image
docker push $AWS_LAMBDA_CONTAINER_REPO.dkr.ecr.$AWS_LAMBDA_REGION.amazonaws.com/weather-collection
