docker_build(
    "sample-img",
    "./sample",
)


k8s_yaml('k8s/sample/sample.yaml')

k8s_resource(workload='sample-rollout')

