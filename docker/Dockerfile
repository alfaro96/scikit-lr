FROM python:3.6-slim

RUN apt-get update; \
    apt-get install -y git gcc g++ make

COPY requirements.txt /req/

RUN pip install --upgrade pip; \
    pip install -r /req/requirements.txt

# Running the container in root mode is
# not recommended for security purposes
ARG USERNAME=scikit-lr
ENV HOME /home/$USERNAME
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
RUN mkdir -p $HOME/workspace
WORKDIR $HOME/workspace

CMD /bin/bash
