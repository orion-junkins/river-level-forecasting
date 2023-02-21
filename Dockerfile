# 1 - set base image
FROM python:3.10
# 2 - set the working directory
WORKDIR /opt/app
# 3 - copy files to the working directory
COPY . .
# 4 - install dependencies
RUN pip install -r requirements.txt
RUN pip install -e /opt/app/.

# 5 - command that runs when container starts
CMD ["python", "/opt/app/scripts/upload/weather/current_meteo_to_aws.py"]