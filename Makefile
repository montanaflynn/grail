.PHONY: all fmt fmt-check lint test

all: fmt lint test

fmt:
	go fmt ./...

fmt-check:
	@fmt_out=$$(gofmt -l .); \
	if [ -n "$$fmt_out" ]; then \
		echo "gofmt found issues:"; \
		echo "$$fmt_out"; \
		exit 1; \
	fi

lint:
	go vet ./...

test:
	go test ./...

