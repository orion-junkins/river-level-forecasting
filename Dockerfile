FROM continuumio/miniconda
WORKDIR /usr/src/app
COPY ./ ./
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "river-level", "/bin/bash", "-c"]

RUN pip install -e .