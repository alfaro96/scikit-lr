FROM python:3.6-slim

# Install system requirements
RUN apt-get update; \
    apt-get install -y git gcc g++ make r-base

# Install Python requirements
COPY requirements/*.txt /req/
RUN pip install --upgrade pip; \
    pip install -r /req/benchmarks.txt \
                -r /req/docs.txt \
                -r /req/install.txt \
                -r /req/style.txt \
                -r /req/tests.txt

# Install R requirements
COPY commands/* /bin/
RUN chmod +x /bin/pip-r
RUN pip-r /req/analysis.txt

# Run the container in user mode
ARG USERNAME=scikit-lr
ENV HOME /home/$USERNAME
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
RUN mkdir -p $HOME/workspace
WORKDIR $HOME/workspace

CMD /bin/bash
