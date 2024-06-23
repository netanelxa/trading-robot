# -*- mode: Python -*-
# vi:si:et:sw=2:sts=2:ts=2

# install argo
local('echo local')

#######################
# Install sample app: #
#######################
local_resource(
  'sample-bin',
  'echo running sample-bin',
  deps=[
    './sample' # will watch changed files here
  ],
  labels=['build-stuff']
)

docker_build(
    "sample",   #// tag for image
    "./sample", # // context? dir?
    dockerfile = 'Dockerfile' # // this is the default dockerfile name can be removed
)

#^^ this should build image names "sample"
sampleYaml=kustomize('kubernetes/sample')
k8s_yaml(sampleYaml)

#^^ this should become a pod:
k8s_resource(
  workload='sample',
  labels=['microservice-grp-A'],
  resource_deps = [
    'sample-bin',
  ]
)

