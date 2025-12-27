#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {message}", file=sys.stderr, flush=True)


def resolve_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(script_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out).resolve()
    except Exception:
        pass
    return script_dir.parent.parent.resolve()


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pidfile(path: Path) -> int | None:
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def split_cgi_response(raw: bytes) -> Tuple[int, Dict[str, str], bytes]:
    for sep in (b"\r\n\r\n", b"\n\n"):
        idx = raw.find(sep)
        if idx == -1:
            continue
        head = raw[:idx].decode("iso-8859-1", errors="replace")
        body = raw[idx + len(sep) :]
        headers: Dict[str, str] = {}
        status_code = 200
        for line in head.splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            name = name.strip()
            value = value.strip()
            if name.lower() == "status":
                try:
                    status_code = int(value.split()[0])
                except Exception:
                    status_code = 200
            else:
                headers[name] = value
        return status_code, headers, body

    return 200, {"Content-Type": "application/octet-stream"}, raw


class OverpassHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        repo_root: Path,
        interpreter_path: Path,
        db_dir: Path,
        socket_dir_env: str,
        dispatcher_pidfile: Path,
        max_body_bytes: int,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.repo_root = repo_root
        self.interpreter_path = interpreter_path
        self.db_dir = db_dir
        self.socket_dir_env = socket_dir_env
        self.dispatcher_pidfile = dispatcher_pidfile
        self.max_body_bytes = max_body_bytes


class Handler(BaseHTTPRequestHandler):
    server: OverpassHTTPServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        log(f"http {self.address_string()} - {fmt % args}")

    def _send_plain(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _require_dispatcher(self) -> bool:
        pid = read_pidfile(self.server.dispatcher_pidfile)
        if pid is None:
            self._send_plain(
                503,
                f"Overpass dispatcher not running (missing pidfile: {self.server.dispatcher_pidfile})\n",
            )
            return False
        if not is_pid_running(pid):
            self._send_plain(
                503,
                f"Overpass dispatcher not running (stale pid {pid} in {self.server.dispatcher_pidfile})\n",
            )
            return False
        return True

    def do_POST(self) -> None:
        if self.path != "/api/interpreter":
            self._send_plain(404, "Not Found\n")
            return

        if not self._require_dispatcher():
            return

        if not self.server.interpreter_path.exists():
            self._send_plain(500, f"Missing interpreter: {self.server.interpreter_path}\n")
            return
        if not os.access(self.server.interpreter_path, os.X_OK):
            self._send_plain(500, f"Interpreter not executable: {self.server.interpreter_path}\n")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/x-www-form-urlencoded"):
            self._send_plain(
                415,
                "Unsupported Content-Type. Use application/x-www-form-urlencoded with field 'data'.\n",
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_plain(400, "Invalid Content-Length\n")
            return
        if length <= 0:
            self._send_plain(400, "Missing/empty request body\n")
            return
        if length > self.server.max_body_bytes:
            self._send_plain(413, f"Request too large (max {self.server.max_body_bytes} bytes)\n")
            return

        body = self.rfile.read(length)
        if b"data=" not in body:
            self._send_plain(400, "Missing form field 'data'\n")
            return

        env = os.environ.copy()
        env.update(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(len(body)),
                "CONTENT_TYPE": content_type,
                "QUERY_STRING": "",
                "SCRIPT_NAME": "/api/interpreter",
                "SERVER_NAME": "127.0.0.1",
                "SERVER_PORT": str(self.server.server_address[1]),
                "SERVER_PROTOCOL": "HTTP/1.1",
                "GATEWAY_INTERFACE": "CGI/1.1",
                "REMOTE_ADDR": self.client_address[0],
                "OVERPASS_DB_DIR": str(self.server.db_dir),
                # AF_UNIX sockets have a short path limit; use a repo-relative socket dir.
                "OVERPASS_SOCKET_DIR": self.server.socket_dir_env,
            }
        )

        try:
            proc = subprocess.run(
                [str(self.server.interpreter_path)],
                input=body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=600,
                cwd=str(self.server.repo_root),
            )
        except subprocess.TimeoutExpired:
            self._send_plain(504, "Upstream interpreter timed out\n")
            return
        except Exception as e:
            self._send_plain(502, f"Failed to run interpreter: {e}\n")
            return

        if proc.returncode != 0 and not proc.stdout:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            self._send_plain(502, f"Interpreter failed (exit {proc.returncode}):\n{stderr}\n")
            return

        status_code, headers, payload = split_cgi_response(proc.stdout)
        self.send_response(status_code)

        sent_ct = False
        for name, value in headers.items():
            lname = name.lower()
            if lname in {"status", "transfer-encoding", "content-length"}:
                continue
            if lname == "content-type":
                sent_ct = True
            self.send_header(name, value)
        if not sent_ct:
            self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            if not self._require_dispatcher():
                return
            self._send_plain(200, "ok\n")
            return
        self._send_plain(404, "Not Found\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Overpass /api/interpreter wrapper (no nginx).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--max-body-bytes", default=16 * 1024 * 1024, type=int)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    interpreter = repo_root / "geodata" / "amensity" / "vendor" / "overpass" / "install" / "cgi-bin" / "interpreter"
    db_dir = repo_root / "geodata" / "overpass" / "db"
    socket_dir_abs = repo_root / "geodata" / "overpass" / "socket"
    socket_dir_abs.mkdir(parents=True, exist_ok=True)

    socket_dir_env = os.environ.get("OVERPASS_SOCKET_DIR", "").strip()
    if socket_dir_env and not socket_dir_env.endswith("/"):
        socket_dir_env += "/"
    if not socket_dir_env:
        # Overpass uses AF_UNIX sockets with a short path limit; deep repo paths can exceed it.
        # Use a stable /tmp symlink that points back into the repo.
        import hashlib

        digest = hashlib.sha1(str(repo_root).encode("utf-8")).hexdigest()[:10]
        socket_link = Path("/tmp") / f"overpass_socket_{digest}"
        if socket_link.exists() or socket_link.is_symlink():
            if socket_link.is_symlink():
                try:
                    target = socket_link.readlink()
                except Exception:
                    target = None
                if target != socket_dir_abs:
                    socket_link.unlink(missing_ok=True)
                    socket_link.symlink_to(socket_dir_abs)
            else:
                log(f"Socket link path exists and is not a symlink: {socket_link}")
                return 1
        else:
            socket_link.symlink_to(socket_dir_abs)
        socket_dir_env = str(socket_link) + "/"
    dispatcher_pidfile = repo_root / "geodata" / "overpass" / "dispatcher.pid"

    log(f"Repo root: {repo_root}")
    log(f"Interpreter: {interpreter}")
    log(f"DB dir: {db_dir}")
    log(f"Dispatcher pidfile: {dispatcher_pidfile}")
    log(f"Listening on http://{args.host}:{args.port}")

    httpd = OverpassHTTPServer(
        (args.host, args.port),
        Handler,
        repo_root=repo_root,
        interpreter_path=interpreter,
        db_dir=db_dir,
        socket_dir_env=socket_dir_env,
        dispatcher_pidfile=dispatcher_pidfile,
        max_body_bytes=args.max_body_bytes,
    )

    def _handle_term(signum: int, _frame) -> None:
        log(f"Received signal {signum}; shutting down")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
