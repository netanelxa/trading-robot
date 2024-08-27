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
kind load docker-image -n trader-cluster redis:6.2-alpine

#dnsutils:
docker pull registry.k8s.io/e2e-test-images/jessie-dnsutils:1.3
 kind load docker-image -n trader-cluster registry.k8s.io/e2e-test-images/jessie-dnsutils:1.3
```

# trubleshoot:
```
k get nodes -A # are not ready
k describe nodes -A
k config use-context kind-trader-cluster
ctlptl get -o json | jq -r '.items[] | select(.registry == "ctlptl-registry") | .status.localRegistryHosting.host'
```

