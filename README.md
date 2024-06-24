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

# trubleshoot:

1.
if stuck in downloading image phase:
"Ensuring node image (kindest/node:v1.27.3) 🖼"
then just run the :
docker pull kindest/node:v1.27.3

2. if 
k get nodes -A # are not ready
#try getting the error with:
k describe nodes -A

if it gives cni plugin not initialized"
# this removes all images beware:
https://stackoverflow.com/questions/69552636/cannot-launch-docker-desktop-for-mac



k config use-context kind-trader-cluster
ctlptl get -o json | jq -r '.items[] | select(.registry == "ctlptl-registry") | .status.localRegistryHosting.host'
