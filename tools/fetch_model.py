#!/usr/bin/env python3
"""Chunked parallel downloader for the MiniMax-H3 full teacher checkpoint.

Why this exists instead of `curl` / `huggingface_hub`:
  * IPv6 is black-holed on this machine: TLS handshake succeeds but large
    response bodies stall forever. Every lookup is pinned to AF_INET.
  * The sandbox transparent proxy (127.0.0.1:58381) and Clash (7890) both
    stall on large bodies too, so proxy env vars are dropped explicitly.
  * A single connection to the ModelScope CDN measures ~0.6 MB/s, so the
    file is split into independent byte ranges served by N connections.
  * Resumable: progress is checkpointed per chunk into a JSON sidecar.

Usage:
  python tools/fetch_model.py --list
  python tools/fetch_model.py --model ref2va_full_int8 --connections 32
  python tools/fetch_model.py --model ref2va_full_int8 --resume
  python tools/fetch_model.py --model ref2va_full_int8 --verify-sha256
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- network hardening: must happen before any socket is created -------------
_ORIG_GETADDRINFO = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return [
        r for r in _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
        if r[0] == socket.AF_INET
    ]


socket.getaddrinfo = _ipv4_only
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "ALL_PROXY"):
    os.environ.pop(_v, None)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) H3-ExactAdaLN-Fetcher/1.0"
_WRITE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()

# (repo-relative path, expected byte size)
MODELS = {
    "ref2va_full_int8": (
        "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
        34038894550,
    ),
    "ref2va_pruned_int8": (
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        20970379616,
    ),
    "ref2va_full_bf16": (
        "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        66280487368,
    ),
    "ref2va_pruned_bf16": (
        "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
        40225724176,
    ),
    "fl2va_full_int8": (
        "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
        34038894550,
    ),
    "ref2va_turbo4_lora": (
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        1956193000,
    ),
    "fl2va_pruned_int8": (
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        20970379616,
    ),
    "fl2va_turbo8_lora": (
        "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        1956193000,
    ),
}

SOURCE_TEMPLATES = [
    "https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/{path}",
    "https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main/{path}",
]


def log(msg):
    print("[fetch] {}".format(msg), flush=True)


def probe(url, timeout=30):
    """Return (total_size, final_url) using a 1-byte ranged request."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cr = r.headers.get("Content-Range")
        total = int(cr.rsplit("/", 1)[1]) if cr else None
        return total, r.geturl()


def resolve_source(path, source_index):
    urls = [t.format(path=path) for t in SOURCE_TEMPLATES]
    order = urls[source_index:] + urls[:source_index]
    last = None
    for url in order:
        try:
            total, final = probe(url)
            if total:
                log("source OK: {} -> {} bytes (cdn: {})".format(
                    url.split("/")[2], total, final.split("/")[2]))
                # Keep the resolve URL, not the CDN URL: CDN links can carry a
                # short-lived signature that expires during a multi-hour fetch.
                # Every ranged request re-resolves to a fresh CDN URL.
                return url, total
        except Exception as e:
            last = "{}: {}".format(type(e).__name__, str(e)[:100])
            log("source failed {}: {}".format(url.split("/")[2], last))
    raise RuntimeError("no reachable source: {}".format(last))


class Downloader:
    def __init__(self, dst, url, total, chunk_bytes, progress_path):
        self.dst = dst
        self.url = url
        self.total = total
        self.chunk_bytes = chunk_bytes
        self.progress_path = progress_path
        self.nchunks = (total + chunk_bytes - 1) // chunk_bytes
        self.done = [0] * self.nchunks
        self.bytes_total = 0
        self.start = time.time()
        self.failed = []
        self._load_progress()
        self._prepare_file()

    def _load_progress(self):
        if not os.path.exists(self.progress_path):
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                p = json.load(f)
            if p.get("total") == self.total and p.get("chunk_bytes") == self.chunk_bytes:
                saved = p.get("done", [])
                self.done = [int(saved[i]) if i < len(saved) else 0
                             for i in range(self.nchunks)]
                log("resumed progress from {}".format(self.progress_path))
        except Exception as e:
            log("ignoring bad progress file: {}".format(e))

    def _prepare_file(self):
        os.makedirs(os.path.dirname(self.dst) or ".", exist_ok=True)
        if not os.path.exists(self.dst) or os.path.getsize(self.dst) != self.total:
            with open(self.dst, "wb") as f:
                f.truncate(self.total)
            log("pre-allocated {} ({} bytes)".format(self.dst, self.total))

    def save_progress(self, force=False):
        with _STATE_LOCK:
            with open(self.progress_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"total": self.total, "chunk_bytes": self.chunk_bytes,
                           "done": self.done, "updated": time.time()}, f)
            os.replace(self.progress_path + ".tmp", self.progress_path)

    def bounds(self, idx):
        start = idx * self.chunk_bytes
        end = min(start + self.chunk_bytes, self.total) - 1
        return start, end

    def fetch_chunk(self, idx, retries=6):
        start, end = self.bounds(idx)
        want = end - start + 1
        pos = self.done[idx]
        if pos >= want:
            return 0
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"User-Agent": _UA,
                             "Range": "bytes={}-{}".format(start + pos, end)})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206:
                        raise RuntimeError("server ignored Range (status {})".format(r.status))
                    with open(self.dst, "r+b", buffering=0) as f:
                        while pos < want:
                            buf = r.read(1 << 20)
                            if not buf:
                                break
                            with _WRITE_LOCK:
                                f.seek(start + pos)
                                f.write(buf)
                            pos += len(buf)
                            self.done[idx] = pos
                            with _STATE_LOCK:
                                self.bytes_total += len(buf)
                            if pos % (8 << 20) < (1 << 20):
                                self.save_progress()
                if pos >= want:
                    self.done[idx] = want
                    self.save_progress()
                    return want
            except Exception as e:
                if attempt == retries - 1:
                    self.failed.append((idx, "{}: {}".format(type(e).__name__, str(e)[:120])))
                    return 0
                time.sleep(min(2 ** attempt, 12))
        return 0

    def run(self, connections):
        todo = [i for i in range(self.nchunks) if self.done[i] < (self.bounds(i)[1] - self.bounds(i)[0] + 1)]
        already = sum(self.done)
        log("{} chunks, {} pending, {:.2f} GiB already done".format(
            self.nchunks, len(todo), already / (1 << 30)))
        self.start = time.time()
        self.bytes_total = 0
        stop = threading.Event()

        def reporter():
            while not stop.wait(10):
                done = sum(self.done)
                elapsed = max(time.time() - self.start, 1e-6)
                rate = self.bytes_total / elapsed / 1e6
                remain = (self.total - done) / max(rate * 1e6, 1)
                log("{:.2f}/{:.2f} GiB  {:.2f} MB/s  eta {:.0f} min  failed={}".format(
                    done / (1 << 30), self.total / (1 << 30), rate,
                    remain / 60, len(self.failed)))

        t = threading.Thread(target=reporter, daemon=True)
        t.start()
        try:
            with ThreadPoolExecutor(max_workers=connections) as pool:
                futures = {pool.submit(self.fetch_chunk, i): i for i in todo}
                for fut in as_completed(futures):
                    fut.result()
        finally:
            stop.set()
            self.save_progress(force=True)

        done = sum(self.done)
        log("finished: {}/{} bytes, failed chunks: {}".format(done, self.total, len(self.failed)))
        for idx, err in self.failed[:20]:
            log("  chunk {} {}".format(idx, err))
        return done == self.total and not self.failed


def sha256_file(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ref2va_full_int8", choices=sorted(MODELS))
    ap.add_argument("--out", default=os.environ.get(
        "H3_MODEL_OUT", os.path.join(os.getcwd(), "models", "diffusion_models")))
    ap.add_argument("--connections", type=int, default=32)
    ap.add_argument("--chunk-mib", type=int, default=128)
    ap.add_argument("--source", type=int, default=0, help="0=modelscope, 1=hf-mirror")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify-sha256", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (p, n) in MODELS.items():
            print("{:24s} {:8.2f} GiB  {}".format(k, n / (1 << 30), p))
        return 0

    path, expected = MODELS[args.model]
    dst = os.path.join(args.out, os.path.basename(path))
    progress = dst + ".progress.json"
    url, total = resolve_source(path, args.source)
    if total != expected:
        log("WARNING: remote size {} != catalog size {}".format(total, expected))

    dl = Downloader(dst, url, total, args.chunk_mib << 20, progress)
    ok = dl.run(args.connections)
    if ok:
        log("complete: {}".format(dst))
        if args.verify_sha256:
            log("sha256: {}".format(sha256_file(dst)))
        try:
            os.remove(progress)
        except OSError:
            pass
        return 0
    log("incomplete - rerun the same command to resume")
    return 2


if __name__ == "__main__":
    sys.exit(main())
