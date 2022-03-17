FROM python:3.8.12


# Get build-essentials to ensure gcc compiler up to date and install pandoc for converting tutorials
RUN apt-get update && apt-get -y install g++ gcc libc-dev pandoc

# Explicitly check a pip install on base deps
COPY judgyprophet/ /judgyprophet/judgyprophet/
COPY tests/ /judgyprophet/tests/
COPY tutorials/ /judgyprophet/tutorials/
COPY MANIFEST.in pyproject.toml poetry.lock README.md /judgyprophet/

WORKDIR /judgyprophet/

# Install dev dependencies using poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false \
    && poetry install

# Compile judgyprophet STAN model
RUN poetry run python -c "from judgyprophet import JudgyProphet; JudgyProphet().compile()"

CMD poetry run pytest /judgyprophet/tests/
