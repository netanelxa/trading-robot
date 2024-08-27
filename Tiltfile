load('ext://dotenv', 'dotenv')
dotenv()

# Install argo
local('echo local')

# Build and watch the trading robot application
local_resource(
  'trading-robot-bin',
  'echo running trading-robot-bin',
  deps=[
    './sample'  # Assuming your app code is still in the 'sample' directory
  ],
  labels=['build-stuff']
)

# Build the Docker image
docker_build(
    "netanelxa/trading-robot:latest",
    "./sample",  # Dockerfile location
    network='host'
)

# Create and manage the Alpaca secrets
local_resource(
  'create-alpaca-secrets',
  cmd='kubectl create secret generic alpaca-secrets --from-literal=api-key-id=$APCA_API_KEY_ID --from-literal=api-secret-key=$APCA_API_SECRET_KEY --dry-run=client -o yaml | kubectl apply -f -',
  deps=['.env']
)

# Apply Kubernetes configurations
k8s_yaml(kustomize('kubernetes/sample'))

# Define the Kubernetes resource
k8s_resource(
  workload='trading-robot',  # Updated to match the new deployment name
  labels=['trading-robot'],
  port_forwards=5000,
  resource_deps=[
    'trading-robot-bin',
  ],
)