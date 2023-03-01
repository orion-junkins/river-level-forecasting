ARG APP_DIR="/app"
ARG PYTHON_VERSION="3.8"

FROM python:${PYTHON_VERSION}-slim

ARG APP_DIR

RUN mkdir -p ${APP_DIR}

COPY src ${APP_DIR}/src
COPY data ${APP_DIR}/data
COPY trained_models ${APP_DIR}/trained_models
COPY pyproject.toml ${APP_DIR}/pyproject.toml
COPY containers/inference/lambda/requirements.txt ${APP_DIR}/requirements.txt
COPY containers/inference/lambda/app.py ${APP_DIR}/

RUN pip3 install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r ${APP_DIR}/requirements.txt && pip install ${APP_DIR}
RUN pip3 install --no-cache-dir awslambdaric

WORKDIR ${APP_DIR}

COPY containers/inference/lambda/entry.sh /

RUN chmod 755 /entry.sh
ENTRYPOINT [ "/entry.sh" ]
CMD [ "app.handler" ]
