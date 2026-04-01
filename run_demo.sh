#!/usr/bin/env bash
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

S3_ENDPOINT="http://localhost:8333"
S3_ACCESS_KEY="mlplatform-access-key"
S3_SECRET_KEY="mlplatform-secret-key"

DOCKER_ENV=(
  -e AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
  -e AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
  -e MLFLOW_TRACKING_URI=http://mlflow:5000
  -e MLFLOW_S3_ENDPOINT_URL=http://seaweedfs:8333
)

# Wait until a compose service reports "healthy".
# Usage: wait_healthy <service> <timeout_seconds>
wait_healthy() {
  local svc="$1" timeout="${2:-300}" elapsed=0
  info "Waiting for $svc to become healthy (timeout ${timeout}s)..."
  while true; do
    local status
    status=$(docker compose ps "$svc" --format '{{.Status}}' 2>/dev/null || true)
    if [[ "$status" == *"(healthy)"* ]]; then
      success "$svc is healthy"
      return 0
    fi
    if (( elapsed >= timeout )); then
      fail "$svc did not become healthy within ${timeout}s (status: $status)"
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

# ── Step 1: Build all Docker images ─────────────────────────────────────────

echo ""
info "============================================"
info " Step 1/8: Building Docker images"
info "============================================"
docker compose --profile tools build
success "All images built"

# ── Step 2: Start infrastructure services ────────────────────────────────────

echo ""
info "============================================"
info " Step 2/8: Starting services"
info "============================================"
docker compose up -d
success "Services started"

# ── Step 3: Wait for core services to be healthy ────────────────────────────

echo ""
info "============================================"
info " Step 3/8: Waiting for services to be healthy"
info "============================================"
wait_healthy postgres   120
wait_healthy seaweedfs  120
wait_healthy mlflow     180

# Airflow and JupyterLab are nice-to-have; don't block on them
wait_healthy airflow    300 || warn "Airflow not healthy yet — UI may not be ready"
wait_healthy jupyterlab 120 || warn "JupyterLab not healthy yet"

# ── Step 4: Create S3 buckets ───────────────────────────────────────────────

echo ""
info "============================================"
info " Step 4/8: Creating S3 buckets"
info "============================================"
for bucket in mlplatform-artifacts mlplatform-data; do
  AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
    aws --endpoint-url "$S3_ENDPOINT" s3 mb "s3://$bucket" 2>/dev/null \
    && success "Created bucket $bucket" \
    || success "Bucket $bucket already exists"
done

# ── Step 5: Run training ────────────────────────────────────────────────────

echo ""
info "============================================"
info " Step 5/8: Running ML training pipeline"
info "============================================"
info "This will take a few minutes (Spark + H2O AutoML)..."

TRAIN_LOG=$(mktemp)
docker run --rm \
  --network ml-platform_default \
  -v "$PWD/scripts:/opt/scripts:ro" \
  "${DOCKER_ENV[@]}" \
  --entrypoint python \
  pysparkling:latest \
  /opt/scripts/train.py 2>&1 | tee "$TRAIN_LOG"

RUN_ID=$(grep "MLflow run ID:" "$TRAIN_LOG" | awk '{print $NF}')
BEST_MODEL=$(grep "Best model:" "$TRAIN_LOG" | sed 's/.*Best model: \([^,]*\).*/\1/')
rm -f "$TRAIN_LOG"

if [[ -z "$RUN_ID" ]]; then
  fail "Could not extract MLflow run ID from training output"
fi
success "Training complete  (run_id=$RUN_ID, model=$BEST_MODEL)"

# ── Step 6: Verify training outputs ─────────────────────────────────────────

echo ""
info "============================================"
info " Step 6/8: Verifying training outputs"
info "============================================"

# Check MLflow run
MLFLOW_RUN=$(curl -s "http://localhost:15000/api/2.0/mlflow/runs/get?run_id=$RUN_ID")
STATUS=$(echo "$MLFLOW_RUN" | python3 -c "import sys,json; print(json.load(sys.stdin)['run']['info']['status'])" 2>/dev/null || echo "UNKNOWN")
if [[ "$STATUS" == "FINISHED" ]]; then
  success "MLflow run $RUN_ID status: FINISHED"
else
  fail "MLflow run status: $STATUS (expected FINISHED)"
fi

# Print metrics
echo "$MLFLOW_RUN" | python3 -c "
import sys, json
metrics = {m['key']:m['value'] for m in json.load(sys.stdin)['run']['data']['metrics']}
print(f\"       AUC      = {metrics.get('auc','N/A')}\")
print(f\"       LogLoss  = {metrics.get('logloss','N/A')}\")
"

# Check S3 artifacts
ARTIFACT_COUNT=$(AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  aws --endpoint-url "$S3_ENDPOINT" s3 ls --recursive "s3://mlplatform-artifacts/1/$RUN_ID/" 2>/dev/null | wc -l | tr -d ' ')
success "MLflow artifacts: $ARTIFACT_COUNT file(s) in S3"

DATA_COUNT=$(AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  aws --endpoint-url "$S3_ENDPOINT" s3 ls --recursive "s3://mlplatform-data/training/" 2>/dev/null | wc -l | tr -d ' ')
success "Training data: $DATA_COUNT file(s) in S3"

# ── Step 7: Run batch prediction ────────────────────────────────────────────

echo ""
info "============================================"
info " Step 7/8: Running batch prediction"
info "============================================"
info "Using model from run $RUN_ID..."

PREDICT_LOG=$(mktemp)
docker run --rm \
  --network ml-platform_default \
  -v "$PWD/scripts:/opt/scripts:ro" \
  "${DOCKER_ENV[@]}" \
  ml-platform-model-serving:latest \
  /opt/scripts/serve.py --run-id "$RUN_ID" 2>&1 | tee "$PREDICT_LOG"

PRED_ROWS=$(grep "Batch prediction complete:" "$PREDICT_LOG" | awk '{print $4}')
rm -f "$PREDICT_LOG"
success "Batch prediction complete ($PRED_ROWS rows)"

# ── Step 8: Verify batch prediction outputs ──────────────────────────────────

echo ""
info "============================================"
info " Step 8/8: Verifying batch prediction outputs"
info "============================================"

BATCH_COUNT=$(AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  aws --endpoint-url "$S3_ENDPOINT" s3 ls --recursive "s3://mlplatform-data/batch_predictions/" 2>/dev/null | wc -l | tr -d ' ')
success "Batch predictions: $BATCH_COUNT file(s) in S3"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} All steps completed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Training run ID : $RUN_ID"
echo "  Best model      : $BEST_MODEL"
echo ""
echo "  Service URLs:"
echo "    MLflow     : http://localhost:15000"
echo "    Airflow    : http://localhost:18080"
echo "    JupyterLab : http://localhost:8888"
echo ""
echo "  To stop all services:"
echo "    docker compose down"
echo ""
echo "  To stop and remove all data:"
echo "    docker compose down -v"
echo ""
