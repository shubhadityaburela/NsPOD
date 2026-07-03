FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .

RUN mamba env create -f environment.yml && \
    mamba clean -afy

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nspod", "python", "main.py"]