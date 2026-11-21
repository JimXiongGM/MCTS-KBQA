from __future__ import annotations

import asyncio
import atexit
import itertools
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
VRAM_THRESHOLD_MIB = 45 * 1024
ACTOR_PORT = 19000
REWARD_PORT = 29000

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

_STARTED_TMUX_SESSIONS: set[str] = set()


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _read_log_excerpt(path: Path, max_lines: int = 80, max_chars: int = 12000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"(log file not found: {path})"
    except Exception as e:
        return f"(failed to read log file {path}: {e})"

    if not content.strip():
        return f"(log file is empty: {path})"

    lines = content.splitlines()
    excerpt = "\n".join(lines[-max_lines:])
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt


def parse_visible_gpu_ids() -> list[int] | None:
    """Return physical GPU ids from CUDA_VISIBLE_DEVICES, or None for all GPUs."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return None
    if not cvd.strip():
        return []
    return [int(x.strip()) for x in cvd.split(",") if x.strip()]


def detect_num_gpus() -> int:
    visible = parse_visible_gpu_ids()
    if visible is not None:
        return len(visible)
    try:
        p = _run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except FileNotFoundError:
        return 0
    if p.returncode != 0:
        return 0
    lines = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
    return len(lines)


def query_gpu_mem_mib() -> list[int]:
    p = _run_capture(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    )
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {p.stderr}")
    return [int(x.strip()) for x in p.stdout.splitlines() if x.strip()]


# SGLang defaults grpc_port to http_port + 10000, so http_port must stay <= 55535.
_SGLANG_GRPC_PORT_OFFSET = 10000
_BACKEND_PORT_HIGH = 65535 - _SGLANG_GRPC_PORT_OFFSET


def get_random_unused_port(low: int = 20000, high: int = _BACKEND_PORT_HIGH) -> int:
    for _ in range(200):
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find unused port in range")


def tmux_session_exists(name: str) -> bool:
    try:
        p = _run_capture(["tmux", "has-session", "-t", name])
    except FileNotFoundError as e:
        raise RuntimeError(
            "`tmux` command not found. Please install it: sudo apt-get install -y tmux"
        ) from e
    return p.returncode == 0


def kill_tmux_session(name: str):
    _run_capture(["tmux", "kill-session", "-t", name])


@dataclass(frozen=True)
class BackendInstance:
    gpu_id: int
    port: int
    model_name: str
    backend_host: str
    tmux_name: str
    dtype: str
    api_key: str
    mem_fraction_static: float

    @property
    def base_url(self) -> str:
        return f"http://{self.backend_host}:{self.port}"

    @property
    def log_path(self) -> Path:
        return Path(tempfile.gettempdir()) / f"{self.tmux_name}.log"

    def _build_shell_cmd(self) -> str:
        """Build the SGLang launch command using the current Python environment."""
        cmd = [sys.executable, "-m", "sglang.launch_server"]
        cmd += [
            "--model-path",
            self.model_name,
            "--port",
            str(self.port),
            "--host",
            self.backend_host,
            "--dtype",
            self.dtype,
            "--api-key",
            self.api_key,
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--disable-cuda-graph",
        ]
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        return f"CUDA_VISIBLE_DEVICES={shlex.quote(str(self.gpu_id))} {cmd_str}"

    async def start(self, kill_existing: bool):
        if kill_existing and tmux_session_exists(self.tmux_name):
            logger.warning(
                f"Found existing tmux session {self.tmux_name}, killing it first..."
            )
            kill_tmux_session(self.tmux_name)
            await asyncio.sleep(1.0)

        self.log_path.unlink(missing_ok=True)
        shell_cmd = self._build_shell_cmd()
        logged_shell_cmd = (
            f"set +e; ({shell_cmd}) >> {shlex.quote(str(self.log_path))} 2>&1; "
            f"code=$?; echo; "
            f'echo "SGLang exited (code=${{code}}). Pane kept open."; '
            f"exec sleep infinity"
        )
        tmux_cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            self.tmux_name,
            "bash",
            "-lc",
            logged_shell_cmd,
        ]

        logger.info(f"Starting SGLang server gpu={self.gpu_id} port={self.port}")
        logger.debug(f"Shell command: {shell_cmd}")
        logger.debug(f"Tmux log file: {self.log_path}")

        def _run():
            subprocess.run(tmux_cmd, check=True, text=True, capture_output=True)

        try:
            await asyncio.to_thread(_run)
        except FileNotFoundError:
            raise RuntimeError(
                "`tmux` command not found. Please install it: sudo apt-get install -y tmux"
            ) from None

        _STARTED_TMUX_SESSIONS.add(self.tmux_name)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def wait_until_ready(self, client: httpx.AsyncClient, timeout_s: int) -> bool:
        urls = [f"{self.base_url}/ping", f"{self.base_url}/v1/models"]
        headers = self._auth_headers()
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            try:
                for url in urls:
                    r = await client.get(url, timeout=2.0, headers=headers)
                    if r.status_code == 200:
                        logger.success(f"Backend ready: {self.base_url}")
                        return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
        logger.error(f"Backend timed out: {self.base_url}")
        logger.error(
            f"Recent tmux log for {self.tmux_name} ({self.log_path}):\n"
            f"{_read_log_excerpt(self.log_path)}"
        )
        return False

    async def stop(self):
        if self.tmux_name not in _STARTED_TMUX_SESSIONS:
            return
        logger.info(f"Stopping backend tmux={self.tmux_name} ...")

        def _run():
            subprocess.run(
                ["tmux", "kill-session", "-t", self.tmux_name],
                check=False,
                text=True,
                capture_output=True,
            )

        await asyncio.to_thread(_run)
        _STARTED_TMUX_SESSIONS.discard(self.tmux_name)


class RouterState:
    def __init__(self, backends: list[BackendInstance], served_model_name: str):
        self.backends = backends
        self._cycle = itertools.cycle(backends)
        self.served_model_name = served_model_name
        self.http_client: httpx.AsyncClient | None = None

    def next_backend(self) -> BackendInstance:
        return next(self._cycle)


@dataclass
class Args:
    # Launch mode (actor / reward orchestration): set --actor_model
    actor_model: str | None = None
    reward_model: str | None = None

    # Serve mode (router + SGLang backends): set --model_name
    model_name: str | None = None
    start_port: int = ACTOR_PORT
    served_model_name: str | None = None

    router_host: str = "127.0.0.1"
    backend_host: str = "127.0.0.1"

    tmux_prefix: str = "sglang-"
    kill_existing: bool = True
    wait_timeout_s: int = 600

    dtype: str = "bfloat16"
    api_key: str = "kbqa"
    # SGLang --mem-fraction-static; 0.45 means ~45% GPU memory per backend.
    gpu_memory_utilization: float = 0.45

    upstream_connect_timeout_s: float = 10.0
    upstream_timeout_s: float = 300.0

    upstream_retries: int = 2
    upstream_retry_backoff_s: float = 0.5
    upstream_retry_max_backoff_s: float = 5.0

    log_level: str = "INFO"


def create_app(args: Args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        num_gpus = detect_num_gpus()
        if num_gpus <= 0:
            raise RuntimeError("No GPU found (nvidia-smi).")
        logger.info(f"Found {num_gpus} GPUs")
        served_model_name = (args.served_model_name or args.model_name).strip()
        if not served_model_name:
            raise RuntimeError("--model_name/--served_model_name is empty")
        logger.info(f"Model: {args.model_name}")

        visible = parse_visible_gpu_ids()
        backends: list[BackendInstance] = []
        mname = args.model_name.split("/")[-1]
        used_ports: set[int] = {args.start_port}
        for local_id in range(num_gpus):
            # Map local index -> physical id when CUDA_VISIBLE_DEVICES is set.
            gpu_id = visible[local_id] if visible is not None else local_id
            while True:
                port = get_random_unused_port()
                if port not in used_ports:
                    used_ports.add(port)
                    break
            backends.append(
                BackendInstance(
                    gpu_id=gpu_id,
                    port=port,
                    model_name=args.model_name,
                    backend_host=args.backend_host,
                    tmux_name=f"{args.tmux_prefix}{local_id}-{mname}",
                    dtype=args.dtype,
                    api_key=args.api_key,
                    mem_fraction_static=args.gpu_memory_utilization,
                )
            )
            logger.info(f"GPU {gpu_id} (local {local_id}): port={port}")

        await asyncio.gather(
            *(b.start(kill_existing=args.kill_existing) for b in backends)
        )

        async with httpx.AsyncClient() as client:
            ok_list = await asyncio.gather(
                *(
                    b.wait_until_ready(client, timeout_s=args.wait_timeout_s)
                    for b in backends
                )
            )
        if not all(ok_list):
            logger.error("Some backends failed to start. Shutting down...")
            await asyncio.gather(*(b.stop() for b in backends))
            raise RuntimeError("backend startup failed")

        state = RouterState(backends=backends, served_model_name=served_model_name)
        state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                args.upstream_timeout_s, connect=args.upstream_connect_timeout_s
            ),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
        )
        app.state.router = state

        logger.success("All backends are ready.")
        logger.success(f"Router: http://{args.router_host}:{args.start_port}")
        logger.success("Backend pool:")
        for b in backends:
            logger.success(f"  {b.base_url}  (tmux: {b.tmux_name})")

        yield

        logger.info("Shutting down router and backends...")
        try:
            await state.http_client.aclose()
        except Exception:
            pass
        await asyncio.gather(*(b.stop() for b in backends))

    app = FastAPI(title="sglang_embedding_router", version="1.0.0", lifespan=lifespan)

    @app.get("/ping")
    async def ping(request: Request):
        state: RouterState = request.app.state.router
        client = state.http_client

        async def _check_one(b: BackendInstance) -> dict[str, Any]:
            last_err: str | None = None
            headers = b._auth_headers()
            for path in ("/ping", "/v1/models"):
                try:
                    r = await client.get(
                        f"{b.base_url}{path}", timeout=2.0, headers=headers
                    )
                    if r.status_code == 200:
                        return {"base_url": b.base_url, "ok": True, "path": path}
                except Exception as e:
                    last_err = str(e)
            return {"base_url": b.base_url, "ok": False, "error": last_err or "unknown"}

        results = await asyncio.gather(*(_check_one(b) for b in state.backends))
        ok = all(r.get("ok") for r in results)
        return JSONResponse(
            status_code=200 if ok else 503, content={"ok": ok, "backends": results}
        )

    @app.get("/backend-info")
    async def backend_info(request: Request):
        state: RouterState = request.app.state.router
        return {
            "model_name": state.served_model_name,
            "count": len(state.backends),
            "backends": [
                {
                    "gpu_id": b.gpu_id,
                    "port": b.port,
                    "base_url": b.base_url,
                    "tmux": b.tmux_name,
                }
                for b in state.backends
            ],
        }

    @app.api_route(
        "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    async def proxy(full_path: str, request: Request):
        state: RouterState = request.app.state.router
        backend = state.next_backend()
        target_url = f"{backend.base_url}/{full_path}"

        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"-> GPU {backend.gpu_id} (port {backend.port})"
        )

        headers = dict(request.headers)
        headers.pop("host", None)
        headers = {
            k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
        }
        if "authorization" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {backend.api_key}"

        body = await request.body()
        try:
            last_exc: Exception | None = None
            for attempt in range(args.upstream_retries + 1):
                try:
                    req = state.http_client.build_request(
                        request.method,
                        target_url,
                        headers=headers,
                        content=body,
                        params=request.query_params,
                    )
                    resp = await state.http_client.send(req, stream=False)
                    if resp.status_code >= 500 and attempt < args.upstream_retries:
                        await resp.aclose()
                        backoff = min(
                            args.upstream_retry_max_backoff_s,
                            args.upstream_retry_backoff_s * (2**attempt),
                        )
                        logger.warning(
                            f"Upstream {resp.status_code} ({target_url}), retrying in {backoff:.2f}s "
                            f"(attempt {attempt + 1}/{args.upstream_retries})"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    break
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as e:
                    last_exc = e
                    if attempt >= args.upstream_retries:
                        raise
                    backoff = min(
                        args.upstream_retry_max_backoff_s,
                        args.upstream_retry_backoff_s * (2**attempt),
                    )
                    logger.warning(
                        f"Upstream transient error ({target_url}) [{type(e).__name__}] "
                        f"repr={repr(e)}; retrying in {backoff:.2f}s "
                        f"(attempt {attempt + 1}/{args.upstream_retries})"
                    )
                    await asyncio.sleep(backoff)
            else:
                raise last_exc or RuntimeError("unreachable")
        except Exception as e:
            err_type = type(e).__name__
            err_msg = (str(e) or "").strip()
            err_repr = repr(e)
            logger.error(
                f"Upstream Error ({target_url}) [{err_type}] msg={err_msg!r} repr={err_repr}"
            )
            status = 504 if isinstance(e, httpx.TimeoutException) else 502
            return Response(
                status_code=status,
                content=f"Upstream Error [{err_type}]: {err_msg or err_repr}",
            )

        resp_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
        }
        content = resp.content
        await resp.aclose()
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        )

    return app


def _kill_started_tmux_sync():
    for name in list(_STARTED_TMUX_SESSIONS):
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", name],
                check=False,
                text=True,
                capture_output=True,
            )
        except Exception:
            pass
        _STARTED_TMUX_SESSIONS.discard(name)


def _resolve_model_path(label: str, model_path: str) -> Path:
    p = Path(model_path)
    resolved = p if p.is_absolute() else REPO_ROOT / p
    if not resolved.exists():
        raise SystemExit(f"{label} path not found: {model_path} (resolved: {resolved})")
    canonical = resolved.resolve()
    if canonical != resolved:
        print(f"{label}: {model_path} -> {canonical}")
    else:
        print(f"{label}: {model_path}")
    return canonical


def kill_existing_sglang_tmux():
    p = _run_capture(["tmux", "list-sessions", "-F", "#{session_name}"])
    if p.returncode != 0:
        time.sleep(2)
        return
    killed = False
    for session in p.stdout.splitlines():
        session = session.strip()
        if not session.startswith("sglang-"):
            continue
        print(f"Killing tmux session: {session}")
        _run_capture(["tmux", "kill-session", "-t", session])
        killed = True
    if killed:
        print("Waiting 2s for GPU memory to be released...")
    else:
        print("No existing sglang tmux sessions.")
    time.sleep(2)


def _serve_shell_cmd(
    model_name: str, start_port: int, gpu_util: float, cuda_devices: str | None
) -> str:
    env_prefix = f"CUDA_VISIBLE_DEVICES={cuda_devices} " if cuda_devices else ""
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{env_prefix}{shlex.quote(sys.executable)} {shlex.quote(str(THIS_FILE))} "
        f"--model-name {shlex.quote(model_name)} "
        f"--start-port {start_port} "
        f"--gpu-memory-utilization {gpu_util}"
    )


def start_router_tmux(session_name: str, shell_cmd: str):
    wrapped = (
        f"{shell_cmd}; code=$?; wait 2>/dev/null || true; echo; "
        f'echo "Process exited (code=${{code}}). Pane kept open."; exec sleep infinity'
    )
    print(f"tmux new -ds {session_name}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", wrapped],
        check=True,
    )


def launch_actor_reward(actor_model: str, reward_model: str | None):
    """Orchestrate actor (and optional reward) routers in tmux by GPU VRAM layout."""
    _resolve_model_path("actor model", actor_model)
    if reward_model:
        _resolve_model_path("reward model", reward_model)

    if not shutil_which("nvidia-smi"):
        raise SystemExit("nvidia-smi not found")
    if not shutil_which("tmux"):
        raise SystemExit("tmux not found. Install: sudo apt-get install -y tmux")

    mems = query_gpu_mem_mib()
    num_gpus = len(mems)
    if num_gpus <= 0:
        raise SystemExit("no GPU detected")

    if not reward_model:
        print(f"Mode: actor only ({num_gpus} GPU(s))")
        kill_existing_sglang_tmux()
        start_router_tmux(
            "sglang-api-actor",
            _serve_shell_cmd(actor_model, ACTOR_PORT, 0.9, None),
        )
        print("Done. Attach: tmux attach -t sglang-api-actor")
        return

    all_large = all(m > VRAM_THRESHOLD_MIB for m in mems)
    all_small = all(m < VRAM_THRESHOLD_MIB for m in mems)
    print(f"Detected {num_gpus} GPU(s), VRAM (MiB): {mems}")

    if all_large:
        print("Mode: large VRAM (all > 45GB) — reward then actor on all GPUs")
        kill_existing_sglang_tmux()
        start_router_tmux(
            "sglang-api-reward",
            _serve_shell_cmd(reward_model, REWARD_PORT, 0.45, None),
        )
        print("Waiting 2 minutes for reward model to load...")
        time.sleep(120)
        start_router_tmux(
            "sglang-api-actor",
            _serve_shell_cmd(actor_model, ACTOR_PORT, 0.45, None),
        )
        print("Done. Attach: tmux attach -t sglang-api-reward | sglang-api-actor")
        return

    if all_small:
        if num_gpus < 3:
            raise SystemExit(
                f"all GPUs < 45GB but only {num_gpus} GPU(s); need at least 3"
            )
        actor_devices = ",".join(str(i) for i in range(1, num_gpus))
        print(f"Mode: small VRAM — GPU0=reward, GPU[{actor_devices}]=actor")
        kill_existing_sglang_tmux()
        start_router_tmux(
            "sglang-api-reward",
            _serve_shell_cmd(reward_model, REWARD_PORT, 0.9, "0"),
        )
        start_router_tmux(
            "sglang-api-actor",
            _serve_shell_cmd(actor_model, ACTOR_PORT, 0.45, actor_devices),
        )
        print("Done. Attach: tmux attach -t sglang-api-reward | sglang-api-actor")
        return

    raise SystemExit(
        f"mixed GPU VRAM (some >= 45GB, some < 45GB); cannot auto-select layout. "
        f"GPU VRAM (MiB): {mems}, threshold={VRAM_THRESHOLD_MIB}"
    )


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def run_serve(args: Args):
    assert args.model_name is not None
    args.served_model_name = (args.served_model_name or args.model_name).strip()

    if args.start_port <= 0 or args.start_port >= 65535:
        raise SystemExit(f"invalid --start_port: {args.start_port}")
    if not args.served_model_name:
        raise SystemExit("--model_name/--served_model_name is empty")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise SystemExit(
            f"invalid --gpu_memory_utilization: {args.gpu_memory_utilization} "
            "(must be in (0, 1], e.g. 0.45 for 45%)"
        )

    app = create_app(args)
    atexit.register(_kill_started_tmux_sync)

    def _handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    uvicorn.run(app, host=args.router_host, port=args.start_port, log_level="warning")


def main():
    import tyro

    args = tyro.cli(Args, use_underscores=True)

    if args.actor_model:
        launch_actor_reward(args.actor_model, args.reward_model)
        return
    if args.model_name:
        run_serve(args)
        return
    raise SystemExit(
        "need --actor_model (launch) or --model_name (serve). "
        "Example: python scripts/start_sglang.py "
        "--actor_model save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5"
    )


if __name__ == "__main__":
    main()

"""
# Launch actor (+ optional reward) in tmux:
python scripts/start_sglang.py \\
  --actor_model save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5

python scripts/start_sglang.py \\
  --actor_model save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5 \\
  --reward_model meta-llama/Meta-Llama-3.1-8B-Instruct

# Serve mode (usually started by launch above):
CUDA_VISIBLE_DEVICES=0 python scripts/start_sglang.py \\
  --model_name save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5 \\
  --start_port 19000 --gpu_memory_utilization 0.9
"""
