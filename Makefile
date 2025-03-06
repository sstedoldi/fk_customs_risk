IMAGE_NAME = ddt_risk_image
CONTAINER_NAME = ddt_risk_container
PYTHON = python3
SRC_DIR = app
TEST_DIR = tests

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR)

test:
	$(PYTHON) -m pytest $(TEST_DIR)

run:
	$(PYTHON) $(SRC_DIR)/main.py
#	$(PYTHON) $(SRC_DIR)/main_copy.py


# build:
#     docker build -t $(IMAGE_NAME) .

# run:
#     docker run -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# stop:
#     docker stop $(CONTAINER_NAME)

# clean:
#     docker rm $(CONTAINER_NAME)
#     docker rmi $(IMAGE_NAME)