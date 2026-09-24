"""Per-visitor rate limits and security headers.

The free LLM and embedding quotas are the site's scarcest resource; one
script hammering /api/research could spend a day's quota in minutes. Each
client IP gets a sliding window per endpoint. Render terminates TLS and
forwards the client address, which uvicorn --proxy-headers puts in
request.client.
"""
import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# path prefix: (requests, per seconds)
LIMITS = {
    "/api/research": (6, 60),
    "/api/rerank": (30, 60),
    "/api/search": (60, 60),
    "/api/similar": (60, 60),
}

# The one inline script (theme restore) is pinned by its hash in the template.
CSP = ("default-src 'self'; "
       "script-src 'self' 'sha256-{theme}'; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; "
       "img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'")


class Guard(BaseHTTPMiddleware):
    def __init__(self, app, theme_hash):
        super().__init__(app)
        self.csp = CSP.format(theme=theme_hash)
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()
        self.swept = time.monotonic()

    def _allowed(self, key, limit, window):
        now = time.monotonic()
        with self.lock:
            if now - self.swept > 300:  # forget idle visitors so memory stays flat
                for k in [k for k, q in self.hits.items() if not q or now - q[-1] > 600]:
                    del self.hits[k]
                self.swept = now
            q = self.hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    async def dispatch(self, request, call_next):
        path = request.url.path
        for prefix, (limit, window) in LIMITS.items():
            if path.startswith(prefix):
                ip = request.client.host if request.client else "?"
                if not self._allowed((ip, prefix), limit, window):
                    return JSONResponse({"error": "درخواست‌ها زیاد شد. یک دقیقه صبر کنید و دوباره تلاش کنید."},
                                        status_code=429, headers={"Retry-After": str(window)})
                break
        response = await call_next(request)
        h = response.headers
        h["Content-Security-Policy"] = self.csp
        h["X-Content-Type-Options"] = "nosniff"
        h["Referrer-Policy"] = "strict-origin-when-cross-origin"
        h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.headers.get("x-forwarded-proto") == "https":
            h["Strict-Transport-Security"] = "max-age=31536000"
        return response
