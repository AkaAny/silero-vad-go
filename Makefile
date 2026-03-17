.PHONY: test
test:
	go test -v -race -failfast ./...

.PHONY: native
native:
	@if [ -z "$(ONNXRUNTIME_ROOT)" ]; then \
		echo "ONNXRUNTIME_ROOT is required, e.g. make native ONNXRUNTIME_ROOT=/opt/onnxruntime"; \
		exit 1; \
	fi
	cmake -S native -B native/build -DONNXRUNTIME_ROOT="$(ONNXRUNTIME_ROOT)"
	cmake --build native/build --config Release

.PHONY: lint
lint: 
	@if ! [ -x "$$(command -v golangci-lint)" ]; then \
		echo "golangci-lint is not installed. Please see https://github.com/golangci/golangci-lint#install for installation instructions."; \
		exit 1; \
	fi; \

	@echo Running golangci-lint
	golangci-lint run ./...
