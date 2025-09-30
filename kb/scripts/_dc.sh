# Shell: docker compose / docker-compose farkını soyutlar ve ops klasörüne cd eder.
if command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    DC="docker compose"
fi
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)" # .../repo/kb
OPS_DIR="$REPO_DIR/../ops"
cd "$OPS_DIR"
export DC REPO_DIR OPS_DIR