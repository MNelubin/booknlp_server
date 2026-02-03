#!/bin/bash
# BookNLP GPU Service Test Script
# Tests service availability, model state, and processing functionality

set -e

# Configuration
BASE_URL="${BOOKNLP_SERVICE_URL:-http://localhost:8888}"
TEXT="Frodo Baggins was a hobbit who lived in the Shire. He had a friend named Samwise Gamgee."
BOOK_ID="test_$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Test 1: Service availability
test_service_availability() {
    print_header "Test 1: Service Availability"

    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/" 2>/dev/null || echo "000")

    if [ "$response" = "200" ]; then
        print_success "Service is reachable at $BASE_URL"
        curl -s "$BASE_URL/" | jq '.'
    else
        print_error "Service is not reachable (HTTP $response)"
        return 1
    fi
}

# Test 2: Health check
test_health() {
    print_header "Test 2: Health Check"

    response=$(curl -s "$BASE_URL/health")

    if echo "$response" | jq -e '.status == "healthy"' > /dev/null; then
        print_success "Service is healthy"

        # Check CUDA
        cuda_available=$(echo "$response" | jq -r '.cuda_available')
        if [ "$cuda_available" = "true" ]; then
            print_success "CUDA is available"
            gpu_name=$(echo "$response" | jq -r '.gpu_name')
            gpu_count=$(echo "$response" | jq -r '.gpu_count')
            print_info "GPU: $gpu_name (x$gpu_count)"
        else
            print_error "CUDA is not available"
        fi

        # Check model state
        model_loaded=$(echo "$response" | jq -r '.model_loaded')
        if [ "$model_loaded" = "true" ]; then
            print_success "Model is loaded in GPU memory"
            memory_used=$(echo "$response" | jq -r '.gpu_memory_used_mb // "N/A"')
            memory_cached=$(echo "$response" | jq -r '.gpu_memory_cached_mb // "N/A"')
            print_info "GPU Memory: ${memory_used}MB used, ${memory_cached}MB cached"
        else
            print_info "Model is not loaded (will load on first request)"
        fi
    else
        print_error "Health check failed"
        echo "$response" | jq '.'
        return 1
    fi
}

# Test 3: Manual model load
test_model_load() {
    print_header "Test 3: Manual Model Load"

    print_info "Sending request to load model..."
    response=$(curl -s -X POST "$BASE_URL/model/load")

    status=$(echo "$response" | jq -r '.status')

    if [ "$status" = "loaded" ] || [ "$status" = "already_loaded" ]; then
        print_success "$status"
    else
        print_error "Failed to load model"
        echo "$response" | jq '.'
        return 1
    fi
}

# Test 4: Text extraction
test_extraction() {
    print_header "Test 4: Text Extraction"

    print_info "Sending test text for processing..."
    print_info "Text: \"$TEXT\""
    print_info "Book ID: $BOOK_ID"

    response=$(curl -s -X POST "$BASE_URL/extract" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$TEXT\", \"book_id\": \"$BOOK_ID\"}")

    status=$(echo "$response" | jq -r '.status')

    if [ "$status" = "success" ]; then
        print_success "Extraction completed"
        message=$(echo "$response" | jq -r '.message')
        print_info "$message"

        output_dir=$(echo "$response" | jq -r '.output_dir')
        files=$(echo "$response" | jq -r '.files | length')
        print_info "Output directory: $output_dir"
        print_info "Files generated: $files"

        echo "$response" | jq '.files[]' | while read -r file; do
            print_info "  - $file"
        done
    else
        print_error "Extraction failed"
        echo "$response" | jq '.'
        return 1
    fi
}

# Test 5: Get generated files
test_get_files() {
    print_header "Test 5: Get Generated Files"

    response=$(curl -s "$BASE_URL/files/$BOOK_ID")

    if echo "$response" | jq -e '.files' > /dev/null; then
        print_success "Files retrieved for book_id: $BOOK_ID"

        echo "$response" | jq -r '.files[] | "\(.name) - \(.size) bytes"' | while read -r file_info; do
            print_info "  - $file_info"
        done
    else
        print_error "Failed to retrieve files"
        echo "$response" | jq '.'
        return 1
    fi
}

# Test 6: Manual model unload
test_model_unload() {
    print_header "Test 6: Manual Model Unload"

    read -p "Do you want to test model unload? This will free GPU memory. (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Sending request to unload model..."
        response=$(curl -s -X POST "$BASE_URL/model/unload")

        status=$(echo "$response" | jq -r '.status')

        if [ "$status" = "unloaded" ]; then
            print_success "Model unloaded successfully"
        elif [ "$status" = "not_loaded" ]; then
            print_info "Model was not loaded"
        else
            print_error "Failed to unload model"
            echo "$response" | jq '.'
            return 1
        fi
    else
        print_info "Skipping model unload test"
    fi
}

# Main execution
main() {
    print_header "BookNLP GPU Service Test Suite"
    echo "Testing service at: $BASE_URL"
    echo "Started at: $(date)"

    # Run tests
    test_service_availability || exit 1
    test_health || exit 1
    test_model_load || exit 1
    test_extraction || exit 1
    test_get_files || exit 1
    test_model_unload || true

    print_header "All Tests Completed!"
    echo "Finished at: $(date)"
    echo -e "\n${GREEN}Service is working correctly!${NC}\n"
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    print_error "curl is not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    print_error "jq is not installed. Please install jq for JSON parsing."
    exit 1
fi

# Run main function
main
