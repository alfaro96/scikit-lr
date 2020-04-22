# Makefile to simplify repetitive tasks

# Build the docker image from the Dockerfile
build:
	docker build -t alfaro96/scikit-lr:latest .

# Pull the docker image from Docker Hub
pull:
	docker pull alfaro96/scikit-lr:latest

# Run the docker image in a container, mounting the current directory as workspace
run:
	docker run -it -v $(pwd)/:/home/scikit-lr/workspace/ --rm alfaro96/scikit-lr:latest
