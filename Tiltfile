load('ext://dotenv', 'dotenv')
dotenv()

# Install argo
local('echo local')

# Build and watch the trading robot application
local_resource(
  'trading-robot-bin',
  'echo running trading-robot-bin',
  deps=[
    './main_web_service'  # Assuming your app code is still in the 'main_web_service' directory
  ],
  labels=['build-stuff']
)

# Build the Docker image
docker_build(
    "netanelxa/trading-robot:latest",
    "./main_web_service",  # Dockerfile location
    network='host'
)

# Use the pre-built image from Docker Hub
custom_build(
    'netanelxa/ml-service',
    'docker pull netanelxa/ml-service:latest',
    deps=[],  # No local dependencies to watch for changes
    tag='latest'
)

# Push the latest image to Docker Hub after pulling it
local('docker push netanelxa/ml-service:latest')

# Define the Kubernetes resource and expose it with port forwarding
k8s_resource('ml-service', port_forwards='5002:5002')

# Optionally add a post-push action, such as logging or updating configurations
local('echo "Docker image has been pulled and pushed."')

# Create and manage the Alpaca secrets
local_resource(
  'create-alpaca-secrets',
  cmd='kubectl create secret generic alpaca-secrets --from-literal=api-key-id=$APCA_API_KEY_ID --from-literal=api-secret-key=$APCA_API_SECRET_KEY --from-literal=api-key-avantage=$ALPHA_VANTAGE_API_KEY --dry-run=client -o yaml | kubectl apply -f -',
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