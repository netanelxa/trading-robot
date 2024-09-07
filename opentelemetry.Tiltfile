# -*- mode: Python -*-
load("ext://helm_resource", "helm_repo", "helm_resource")

# Currently only namespace definition:
k8s_yaml(kustomize('kubernetes/otlp'))

helm_repo(
    "opentelemetry",
    "https://open-telemetry.github.io/opentelemetry-helm-charts",
    resource_name="opentelemetry-repo",
    labels=["opentelemetry"],
)

helm_resource(
    "opentelemetry-gateway",
    "opentelemetry/opentelemetry-collector",
    labels=["opentelemetry"],
    resource_deps=[
        "opentelemetry-repo",
        #"opentelemetry-selfsigned-cert",
        #"opentelemetry-linode-cert",
    ],
    namespace="opentelemetry",
    release_name="opentelemetry-gateway",
    flags=[
        "--version=0.91.0",
        "-f",
        "../../k8s/values/opentelemetry-gateway/values.yaml",
    ],
    deps=[
        "../../k8s/values/opentelemetry-gateway/values.yaml",
    ],
    port_forwards=[
        port_forward(60003, 8888),
    ],
    links=[
        link("http://localhost:60003/metrics", "/metrics"),
    ],
)

helm_resource(
    "opentelemetry-scraper",
    "opentelemetry/opentelemetry-collector",
    labels=["opentelemetry"],
    resource_deps=["opentelemetry-repo", "opentelemetry-selfsigned-cert"],
    namespace="opentelemetry",
    release_name="opentelemetry-scraper",
    flags=[
        "--version=0.91.0",
        "-f",
        "../../k8s/values/opentelemetry-scraper/values.yaml",
    ],
    deps=[
        "../../k8s/values/opentelemetry-scraper/values.yaml",
    ],
    port_forwards=[
        port_forward(60002, 8888),
    ],
    links=[
        link("http://localhost:60002/metrics", "/metrics"),
    ],
)

helm_resource(
    "opentelemetry-collector",
    "opentelemetry/opentelemetry-collector",
    labels=["opentelemetry"],
    #resource_deps=["opentelemetry-repo", "opentelemetry-selfsigned-cert"],
    namespace="opentelemetry",
    release_name="opentelemetry-collector",
    flags=[
        "--version=0.91.0",
        "-f",
        "kubernetes/otlp/collector-values.yaml",
    ],
    deps=[
        "kubernetes/otlp/collector-values.yaml"
    ],
    port_forwards=[
        port_forward(60001, 8888),
    ],
    links=[
        link("http://localhost:60001/metrics", "/metrics"),
    ],
)
