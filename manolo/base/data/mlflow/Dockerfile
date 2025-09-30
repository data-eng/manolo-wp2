FROM python:3.11-slim-buster
# Install python packages
COPY requirements.txt /tmp
RUN pip install uv
RUN uv pip install -r /tmp/requirements.txt --system
