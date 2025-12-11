CONTAINER_NAME ?= focus-train
CONTAINER_PROJECT_ROOT ?= /workspace/focus-train

.PHONY: install finetune infer eval

install:
	docker exec -it $(CONTAINER_NAME) bash -lc "$(CONTAINER_PROJECT_ROOT)/scripts/install.sh"

finetune:
	docker exec -it $(CONTAINER_NAME) bash -lc "$(CONTAINER_PROJECT_ROOT)/scripts/finetune.sh"

infer:
	docker exec -it $(CONTAINER_NAME) bash -lc "$(CONTAINER_PROJECT_ROOT)/scripts/infer_demo.sh"

eval:
	docker exec -it $(CONTAINER_NAME) bash -lc "$(CONTAINER_PROJECT_ROOT)/scripts/evaluate.sh"
