# Makefile to simplify repetitive tasks

# The tag of the Docker image must
# be the scikit-lr package version
TAG = latest

build:
	docker build -t alfaro96/scikit-lr:$(TAG) .

pull:
	docker pull alfaro96/scikit-lr:$(TAG)

# Run the container allocating a pseudo-TTY (terminal),
# binding mount the current directory as workspace and
# automatically cleaning up the container when stopped
run:
	docker run -it \
			   -v $(PWD)/:/home/scikit-lr/workspace/ \
			   --rm alfaro96/scikit-lr:$(TAG)
