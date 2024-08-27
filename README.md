# trading-robot

what you need to start developing locally:
- ctlptl
- tilt
- kind
- kubectl
- kustomize

kick off dev cluster:
```
make tilt
```

# Alpaca:

you need a free acount in https://app.alpaca.markets/signup for us to fetch market data. add your api keys to .env files (see expected format in .env_example)

# Redis:

to get your image onto dev clusters's registry use:
```
docker pull redis:6.2-alpine 
kind load docker-image  redis:6.2-alpine -n trader-cluster
```

# trubleshoot:
```
k get nodes -A # are not ready
k describe nodes -A
k config use-context kind-trader-cluster
ctlptl get -o json | jq -r '.items[] | select(.registry == "ctlptl-registry") | .status.localRegistryHosting.host'
```

