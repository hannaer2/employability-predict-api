#!/bin/bash
set -e

echo "🔧 Installing system libraries for LightGBM..."
apt-get update -y
apt-get install -y libgomp1 libgfortran5
echo "✅ System libraries installed successfully!"
