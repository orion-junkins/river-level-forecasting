# Building and Deploying Inference Containers to AWS Lambda

## General Architecture

Our current inference architecture using AWS Lambda is built around containers.
In a nutshell, we build a Docker container for each individually trained model.
In that container is the model and the code and environment necessary to run inference on that model.
The container is then deployed to AWS where a Lambda function is scheduled to invoke it daily.
The inference results are placed on S3 as part of the inference process that is located in the container.

## Building the Base Image

All inference model images depend on a base inference image.
That base image must be built locally prior to attempting to deploy a model.
This base image should speed up, and reduce the memory usage of, the actual inference models.

To build the base image, run the following from the repo's root directory:
```
docker build -t inference:base -f containers/inference/lambda/Dockerfile-base .
```

## Deploying an Inference Model

Building and deploying a model has been automated using a shell script.
You will need to have Docker installed and setup as well as AWS's CLI (i.e. the `aws` command must be available).

The following environment variables must be set:

`AWS_LAMBDA_CONTAINER_REPO`: Usually a numeric code that can be found on AWS ECR. It will be the first set of numbers for the repository URI.

`AWS_LAMBDA_REGION`: This is the region that the AWS ECR is located in. This might be something like "us-west-2". This can again be found in the repository URI.

To deploy a model, you must be located in the base directory of the repo and then run the following command:
```
sh containers/inference/lambda/deploy_model.sh {model ID}
```

Where {model ID} is the name of a model that is located in the `trained_models` directory.
That model will be copied to a staging directory located at `containers/inference/lambda/staged_models`.
If that staging directory already exists then it will be deleted and a new one will be created.
Anything located in the staging directory will be permanently deleted.
The staging directory will be deleted again after the script runs.

A docker image will then be built from `Dockerfile-model` that will be tagged with `inference:{model ID}`.
The finished image will get tagged again to be `{AWS URI}/model-inference:{model ID}` and that tagged image will get pushed to AWS.

At this point you will need to setup a lambda function to run daily and to have sufficient permissions to access S3.
Refer to AWS documentation for support with that.

## Lambda Runtime Configuration

We have found it necessary to increase the runtime for the lambda function.
We move it to the maximum of 15 minutes in practice currently, however we find that inference only takes a couple minutes at most.
Additionally, the amount of memory used by inference is between 1 and 2 GB depending on the size of the model.
