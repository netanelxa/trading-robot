# trading-robot

what you need to start developing locally:
1. tilt
2. ctlptl - https://github.com/tilt-l
    to get artifactory host-port
    ctlptl get -o json | jq -r '.items[] | select(.registry == "ctlptl-registry") | .status.localRegistryHosting.host'
3. kind
4. kubectl
5. kustomize


make tilt
