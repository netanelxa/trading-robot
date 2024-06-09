.PHONY: kind
kind:
	ctlptl apply -f ctlptl-config.yaml

.PHONY: tilt
tilt: kind
	tilt up all -f Tiltfile --port 9999


.PHONY: check
check:
	kubectrl get all -A
