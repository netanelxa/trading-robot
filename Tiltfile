load('ext://dotenv', 'dotenv')
dotenv()

# Build and watch the trading robot application
local_resource(
  'trading-robot-bin',
  'echo running trading-robot-bin',
  deps=['./main_web_service'],
  labels=['build-stuff']
)

# Build the Docker image for trading robot
docker_build(
    "netanelxa/trading-robot:latest",
    "./main_web_service",
    network='host'
)

# Use the pre-built image from Docker Hub for ML service
custom_build(
    'netanelxa/ml-service',
    'docker pull netanelxa/ml-service:latest && echo pulled',
    deps=[],
    tag='latest'
)

# Create and manage the Alpaca secrets
local_resource(
  'create-alpaca-secrets',
  cmd='kubectl create secret generic alpaca-secrets --from-literal=api-key-id=$APCA_API_KEY_ID --from-literal=api-secret-key=$APCA_API_SECRET_KEY --from-literal=api-key-avantage=$ALPHA_VANTAGE_API_KEY --dry-run=client -o yaml | kubectl apply -f -',
  deps=['.env']
)

# Apply Kubernetes configurations
k8s_yaml(kustomize('kubernetes/sample'))

# Define the Kubernetes resources
k8s_resource(
  'trading-robot',
  labels=['trading-robot'],
  port_forwards=5000,
  resource_deps=['trading-robot-bin', 'create-alpaca-secrets'],
)

k8s_resource(
  'ml-service',
  port_forwards='5002:5002',
  resource_deps=['create-alpaca-secrets']
)