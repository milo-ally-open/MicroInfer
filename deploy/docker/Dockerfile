FROM golang:1.27-alpine AS build

WORKDIR /src/gateway
COPY src/gateway/go.mod src/gateway/go.sum ./
RUN go mod download
COPY src/gateway/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.24

RUN apk add --no-cache ca-certificates \
    && addgroup -S -g 10001 gateway \
    && adduser -S -D -H -u 10001 -G gateway gateway

WORKDIR /app
COPY --from=build --chown=10001:10001 /out/gateway /usr/local/bin/gateway
COPY --chown=10001:10001 deploy/docker/gateway.yml /etc/microinfer/gateway.yml

USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/gateway"]
CMD ["--config", "/etc/microinfer/gateway.yml"]
