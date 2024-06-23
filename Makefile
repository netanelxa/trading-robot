.PHONY: kind
kind:
	ctlptl apply -f ctlptl-config.yaml

.PHONY: kind-down
kind-down:
	ctlptl delete -f ctlptl-config.yaml

.PHONY: tilt
tilt: kind
	export REGISTRY=$$(ctlptl get -o json | jq -r '.items[] | select(.registry == "ctlptl-registry") | .status.localRegistryHosting.host') && echo $$REGISTRY
	tilt up --port 9999

.PHONY: check
check:
	kubectrl get all -A
