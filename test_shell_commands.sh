#!/bin/bash

echo "Testing KnowProlog JSON-LD import functionality"
echo "================================================"

# Test 1: Show help
echo "1. Testing help command:"
cargo run --release -- --help
echo

# Test 2: Import JSON-LD file
echo "2. Testing JSON-LD file import:"
cargo run --release -- import-jsonld test_data.jsonld
echo

# Test 3: Import JSON-LD content
echo "3. Testing JSON-LD content import:"
cargo run --release -- import-jsonld-content '{
  "@context": {
    "name": "http://example.org/kg/name",
    "Person": "http://example.org/kg/Person"
  },
  "@id": "http://example.org/kg/entity/王五",
  "@type": "Person",
  "name": "王五"
}'
echo

# Test 4: Run interactive mode
echo "4. Testing interactive mode (will exit immediately):"
echo "quit" | cargo run --release -- interactive
echo

echo "All tests completed!"