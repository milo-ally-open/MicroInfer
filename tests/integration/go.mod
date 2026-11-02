module example.com/microinfer/gateway-integration

go 1.23.0

require example.com/videostreamapi/gateway v0.0.0

require gopkg.in/yaml.v3 v3.0.1 // indirect

replace example.com/videostreamapi/gateway => ../../src/gateway
