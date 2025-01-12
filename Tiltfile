# -*- mode: Python -*-
load('ext://dotenv', 'dotenv')
load("ext://helm_resource", "helm_repo", "helm_resource")
dotenv()

# Build and watch the trading robot application
local_resource(
  'trading-robot-bin',
  'echo running trading-robot-bin',
  deps=['./main_web_service'],
  labels=['builds']
)

# Build the Docker image for trading robot
docker_build(
    "netanelxa/trading-robot:latest",
    "./main_web_service",
    network='host',
)

# Use the pre-built image from Docker Hub for ML service
# custom_build(
#     'netanelxa/ml-service',
#     'docker pull netanelxa/ml-service:latest && echo pulled',
#     deps=[],
#     tag='latest'
# )

# Build the Docker image for ML service
docker_build(
    "netanelxa/ml-service:latest",
    "./ml_service",  # Adjust this path to where your ML service code is located
    network='host',
)

# Create and manage the Alpaca secrets
local_resource(
  'create-alpaca-secrets',
  cmd='kubectl create secret generic alpaca-secrets --from-literal=api-key-id=$APCA_API_KEY_ID --from-literal=api-secret-key=$APCA_API_SECRET_KEY --from-literal=api-key-avantage=$ALPHA_VANTAGE_API_KEY --from-literal=tg-bot-token=$TELEGRAM_BOT_TOKEN --from-literal=tg-chat-1=$TELEGRAM_CHAT_ID1   --from-literal=admin-pass=$ADMIN_PASSWORD --from-literal=app-key=$APP_KEY --dry-run=client -o yaml | kubectl apply -f -',
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
  resource_deps=['create-alpaca-secrets'],
  labels=['trading-robot'],
)

##### OTEL #####

helm_repo(
    "open-telemetry",
    "https://open-telemetry.github.io/opentelemetry-helm-charts",
    resource_name="opentelemetry-repo",
    labels=["opentelemetry"],
)

helm_resource(
  "opentelemetry-operator",
  "open-telemetry/opentelemetry-operator",
  namespace="opentelemetry-operator-system",
  flags=[
      '--create-namespace',
      '--namespace=opentelemetry-operator-system',
      '--set=manager.collectorImage.repository=otel/opentelemetry-collector-k8s',
      '--set=admissionWebhooks.certManager.enabled=false',
      '--set=admissionWebhooks.autoGenerateCert.enabled=true',
  ],
  labels=["opentelemetry"],
)

k8s_yaml(kustomize('kubernetes/otel'))

helm_resource(
    "my-opentelemetry-collector",
    "open-telemetry/opentelemetry-collector",
    flags=[
        '--values=kubernetes/otel/values.yaml',
        # moved below to values.yaml above ^
        # '--set=image.repository=otel/opentelemetry-collector-k8s',
        # '--set=mode=deployment',
    ],
      deps=[
          "kubernetes/otel/values.yaml",
      ],
    labels=["opentelemetry"],
)