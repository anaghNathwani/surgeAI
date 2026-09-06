#!/usr/bin/env python3
"""
nathwaniGPT GUI — llama.cpp inference backend + React frontend.

llama-cpp-python serves an OpenAI-compatible API at localhost:8080.
The React frontend connects to it via the Vite proxy — no changes needed there.

Requirements:
    pip install "llama-cpp-python[server]"

    macOS / Apple Silicon (Metal):
    CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python[server]"

Usage:
    python app.py --model /path/to/phi4-mini.gguf
    python app.py --model /path/to/model.gguf --gpu-layers -1
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FRONTEND_DIR = Path(__file__).parent / "frontend"


def _check_llama_cpp() -> None:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        print("ERROR: llama-cpp-python is not installed.")
        print("Install it with:")
        print('  pip install "llama-cpp-python[server]"')
        print("macOS / Apple Silicon:")
        print('  CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python[server]"')
        sys.exit(1)


def _start_llama_server(
    model_path: Path,
    host: str,
    port: int,
    n_ctx: int,
    n_gpu_layers: int,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model",        str(model_path),
        "--host",         host,
        "--port",         str(port),
        "--n_ctx",        str(n_ctx),
        "--n_gpu_layers", str(n_gpu_layers),
    ]
    # Stream server output to the terminal so errors are visible
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def _start_vite() -> subprocess.Popen:
    if not (FRONTEND_DIR / "node_modules").exists():
        print("Installing frontend dependencies…")
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)

    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="nathwaniGPT GUI")
    parser.add_argument("--model", default=None,
                        help="Path to a GGUF model file. Omit to auto-detect from weights/")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--ctx",        type=int, default=4096,
                        help="Context window tokens (default: 4096)")
    parser.add_argument("--gpu-layers", type=int, default=0,
                        help="Layers to offload to GPU. -1 = all (default: 0 = CPU)")
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model).expanduser().resolve()
    else:
        # Auto-detect: find the first .gguf in weights/, prefer Q4_K_M
        weights_dir = ROOT / "weights"
        candidates = sorted(weights_dir.rglob("*.gguf"))
        preferred   = [p for p in candidates if "Q4_K_M" in p.name]
        model_path  = (preferred or candidates or [None])[0]

    if model_path is None or not model_path.exists():
        print("ERROR: no GGUF model found.")
        print("Download one first (run from the project root):")
        print(f"  python {ROOT / 'scripts' / 'download_gguf.py'}")
        print("Or pass the path directly:")
        print("  python GUI/app.py --model /path/to/model.gguf")
        sys.exit(1)

    _check_llama_cpp()

    print(f"Starting llama.cpp server  →  {model_path.name}")
    llama_proc = _start_llama_server(
        model_path, args.host, args.port, args.ctx, args.gpu_layers
    )

    print("Starting Vite dev server…")
    vite_proc = _start_vite()

    # Give the llama server a moment to bind before opening the browser
    time.sleep(2.5)
    url = "http://localhost:5173"
    webbrowser.open(url)
    print(f"\nOpened {url}  —  Ctrl+C to stop.\n")

    try:
        llama_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        llama_proc.terminate()
        vite_proc.terminate()


if __name__ == "__main__":
    main()
