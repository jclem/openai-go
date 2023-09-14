.PHONY: test integration lint

test:
	@go test -v ./...

integration:
	@go test -v -tags=integration ./...

lint:
	@golangci-lint run