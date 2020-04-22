# Makefile to simplify repetitive tasks

# The tag of the image must be
# the scikit-lr package version
TAG = latest

build:
	docker build -t alfaro96/scikit-lr:$(TAG) .

pull:
	docker pull alfaro96/scikit-lr:$(TAG)

# Run the container allocating a pseudo-TTY (terminal),
# binding mount the current directory as workspace and
# automatically removing the container when it exits
run:
	docker run -it \
			   -v $(PWD)/:/home/scikit-lr/workspace/ \
			   --rm alfaro96/scikit-lr:$(TAG)
