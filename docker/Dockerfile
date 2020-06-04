FROM python:3.6-slim

# Install system dependencies
RUN apt-get update; \
    apt-get install -y git gcc g++ make

# Run the container in user mode
ARG USERNAME=scikit-lr
ENV HOME /home/$USERNAME
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME

# Move to the workspace
RUN mkdir -p $HOME/workspace
WORKDIR $HOME/workspace

# Add the path to the packages
ENV PATH="$PATH:$HOME/.local/bin"

# Install Python requirements
COPY requirements.txt /req/
RUN pip install --upgrade pip; \
    pip install -r /req/requirements.txt

# Install from source
ARG SOURCE_COMMIT
RUN echo "Installing commit $SOURCE_COMMIT."
RUN pip install git+git://github.com/alfaro96/scikit-lr.git@$SOURCE_COMMIT

# Expose the port to listen
EXPOSE 8888
