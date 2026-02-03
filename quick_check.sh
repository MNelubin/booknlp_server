#!/bin/bash
# Quick health check for BookNLP GPU Service
# Fast check without running extraction

BASE_URL="${BOOKNLP_SERVICE_URL:-http://localhost:9999}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Checking BookNLP GPU Service at $BASE_URL"
echo

# Check service availability
status_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")

if [ "$status_code" != "200" ]; then
    echo -e "${RED}❌ Service is not reachable (HTTP $status_code)${NC}"
    exit 1
fi

# Get health info
health=$(curl -s "$BASE_URL/health")

# Parse JSON
service_status=$(echo "$health" | jq -r '.status')
cuda_available=$(echo "$health" | jq -r '.cuda_available')
model_loaded=$(echo "$health" | jq -r '.model_loaded')
gpu_name=$(echo "$health" | jq -r '.gpu_name')

# Display results
if [ "$service_status" = "healthy" ]; then
    echo -e "${GREEN}✓ Service Status:${NC} $service_status"
else
    echo -e "${RED}✗ Service Status:${NC} $service_status"
fi

if [ "$cuda_available" = "true" ]; then
    echo -e "${GREEN}✓ CUDA:${NC} Available"
    echo -e "${GREEN}✓ GPU:${NC} $gpu_name"
else
    echo -e "${RED}✗ CUDA:${NC} Not available"
fi

if [ "$model_loaded" = "true" ]; then
    memory_used=$(echo "$health" | jq -r '.gpu_memory_used_mb // "N/A"')
    echo -e "${GREEN}✓ Model:${NC} Loaded (${memory_used}MB)"
else
    echo -e "${YELLOW}○ Model:${NC} Not loaded (lazy loading enabled)"
fi

echo
echo -e "${GREEN}✅ Service is ready!${NC}"
