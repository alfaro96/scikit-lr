# Makefile to simplify repetitive tasks

# Substitute this value by the release version
# (use the "latest" tag for the master branch)
TAG = latest

build:
	docker build -t alfaro96/scikit-lr:$(TAG) .

pull:
	docker pull alfaro96/scikit-lr:$(TAG)

run:
	docker run -it \
			   -v $(PWD)/:/home/scikit-lr/workspace/ \
			   --rm alfaro96/scikit-lr:$(TAG)
