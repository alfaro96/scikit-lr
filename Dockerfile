FROM python:3.6-slim

# Copy the requirements
COPY requirements/*.txt /req/

# Concatenate the requirements file to obtain a new one
# with the whole list
RUN cat /req/*.txt > /req/requirements.txt

# Update the repositories
RUN apt-get update

# Install gcc and g++ version for Cython
RUN apt-get install -y gcc g++ make

# Upgrade pip and install the requirements
RUN pip install --upgrade pip; \
    pip install -r /req/requirements.txt

# Add the user
RUN     useradd -ms /bin/bash plr
USER    plr
RUN     mkdir -p /home/plr/workspace
WORKDIR /home/plr/workspace

# Set the PYTHONPATH to know where the packages are installed
ENV PYTHONPATH /home/plr/.local

# Run the command line
CMD ["/bin/bash"]
