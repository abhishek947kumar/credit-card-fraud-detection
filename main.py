"""
Main Entrypoint for the Credit Card Fraud Detection Platform.
Runs FastAPI server with Uvicorn.
"""
import argparse
import sys
import uvicorn
from src.train import BlendedFraudEnsemble

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Machine Learning Platform")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 70)
    print("  [+] CREDIT CARD FRAUD DETECTION AI PLATFORM")
    print("  Real-time Machine Learning Intelligence for Financial Services")
    print("=" * 70)
    print(f"  Web Dashboard:   http://{args.host}:{args.port}")
    print(f"  Interactive API: http://{args.host}:{args.port}/docs")
    print("  Model Engine:    Active (XGBoost / Random Forest / Isolation Forest)")
    print("=" * 70 + "\n")

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
