import os
import threading
import time
import tomllib
from functools import cache
from itertools import count
from pathlib import Path

import httpx
from loguru import logger

# Use the configured MCP endpoints by default and honor explicit legacy HTTP settings.
with (Path(__file__).resolve().parents[1] / "config" / "kb.toml").open("rb") as stream:
    _MCP_URLS = tomllib.load(stream)["mcp"]
_MCP_PROTOCOL_VERSION = "2025-11-25"
_MCP_REQUEST_IDS = count(1)
_GATEWAY = os.environ.get("KB_GATEWAY_BASE")
_TOKEN = os.environ.get("KB_GATEWAY_TOKEN")
_GATEWAY_ROUTES = {"fb": "freebase", "kqapro": "wikidata"}
_INTERACTIVE_API_BASE = {
    "fb": os.environ.get("KB_FB_API_BASE"),
    "kqapro": os.environ.get("KB_KQAPRO_API_BASE"),
}

# Add authorization when a token is present and share HTTP connections across threads.
_HTTP_HEADERS = {}
if _TOKEN:
    _HTTP_HEADERS["Authorization"] = f"Bearer {_TOKEN}"
_HTTP_CLIENT = httpx.Client(headers=_HTTP_HEADERS)
# Connect directly to the configured MCP endpoints so local requests bypass system proxies.
_MCP_HTTP_CLIENT = httpx.Client(headers=_HTTP_HEADERS, trust_env=False)
_BASE_URL_LOGGED = False
_BASE_URL_LOG_LOCK = threading.Lock()

_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_S = 0.5
_RETRY_MAX_BACKOFF_S = 5.0
_RETRYABLE_HTTP_STATUS = {429, 502, 503, 504}
_DEFAULT_TOOL_TIMEOUT_S = 400.0
_SEARCH_GRAPH_PATTERNS_TIMEOUT_S = 400.0

_TOOL_PATHS = {
    "SearchNodes": "/search_nodes",
    "SearchTypes": "/search_types",
    "SearchGraphPatterns": "/search_graph_patterns",
    "ExecuteSPARQL": "/execute_sparql",
}


def _base_url(db: str) -> str:
    """Select the HTTP endpoint for the requested knowledge graph."""
    if db not in _GATEWAY_ROUTES:
        raise ValueError(f"db: {db} not supported.")
    if legacy_base := _INTERACTIVE_API_BASE[db]:
        return f"{legacy_base.rstrip('/')}/{db}"
    if _GATEWAY:
        return f"{_GATEWAY.rstrip('/')}/{_GATEWAY_ROUTES[db]}"
    return _MCP_URLS[db]


def _mcp_request(url, method, params=None, *, version=None, timeout):
    """Send a request or notification to the stateless MCP server and read any JSON result."""
    headers = {"Accept": "application/json, text/event-stream"}
    if version:
        headers["MCP-Protocol-Version"] = version
    payload = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload.update(id=next(_MCP_REQUEST_IDS), params=params)
    response = _MCP_HTTP_CLIENT.post(
        url, json=payload, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    if params is None:
        return
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"MCP {method} failed at {url}: {data['error']['message']}")
    return data["result"]


def _log_base_url_once() -> None:
    global _BASE_URL_LOGGED
    with _BASE_URL_LOG_LOCK:
        if _BASE_URL_LOGGED:
            return
        _BASE_URL_LOGGED = True
        logger.warning(f"KBQA Freebase API: {_base_url('fb')}")
        logger.warning(f"KBQA Wikidata API: {_base_url('kqapro')}")


def _is_retryable_request_error(exc: httpx.HTTPError) -> bool:
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.ReadError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_HTTP_STATUS
    return False


def _is_timeout_like_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code == 504
    return False


def _format_action_content(name, arguments):
    args = arguments or {}
    if name == "SearchNodes":
        q = args.get("query", "")
        n = args.get("n_results", 10)
        omit = args.get("omit_length")
        s = f"SearchNodes({q!r}, n_results={n}"
        if omit is not None:
            s += f", omit_length={omit}"
        return s + ")"
    if name == "SearchTypes":
        q = args.get("query", "")
        n = args.get("n_results", 10)
        omit = args.get("omit_length")
        s = f"SearchTypes({q!r}, n_results={n}"
        if omit is not None:
            s += f", omit_length={omit}"
        return s + ")"
    if name == "SearchGraphPatterns":
        sparql = args.get("sparql", "")
        semantic = args.get("semantic")
        topN = args.get("topN_return", 10)
        s = f"SearchGraphPatterns({sparql!r}"
        if semantic:
            s += f", semantic={semantic!r}"
        return f"{s}, topN_return={topN})"
    if name == "ExecuteSPARQL":
        sparql = args.get("sparql", "")
        str_mode = args.get("str_mode", True)
        return f"ExecuteSPARQL({sparql!r}, str_mode={str_mode})"
    return f"{name}({args!r})"


def _call_api_tool(db, name, arguments, timeout=600.0, *, mcp_version=None):
    """Call a tool over MCP or legacy HTTP and retry transient HTTP failures."""
    path = _TOOL_PATHS.get(name)
    if path is None:
        raise ValueError(f"tool: {name} not supported.")
    legacy_api = bool(_INTERACTIVE_API_BASE[db])
    payload = arguments or {}
    if legacy_api:
        path = f"/{name}"
        payload_args = {
            "data": {
                key: value
                for key, value in payload.items()
                if value is not None and key != "omit_length"
            }
        }
    else:
        payload_args = {"json": payload}
    url = _base_url(db)
    if not mcp_version:
        url += path
    last_exc = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            if mcp_version:
                result = _mcp_request(
                    url,
                    "tools/call",
                    {"name": name, "arguments": payload},
                    version=mcp_version,
                    timeout=timeout,
                )
                content = "\n".join(
                    part["text"] for part in result["content"] if part["type"] == "text"
                )
                return f"Error: {content}" if result.get("isError") else content
            resp = _HTTP_CLIENT.post(
                url,
                timeout=timeout,
                **payload_args,
            )
            resp.raise_for_status()
            data = resp.json()
            if legacy_api:
                return str(data)
            return str((data or {}).get("content") or "")
        except httpx.HTTPError as exc:
            last_exc = exc
            action_content = _format_action_content(name, arguments)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt >= _MAX_ATTEMPTS - 1 or not _is_retryable_request_error(exc):
                logger.error(
                    f"KB API failed for {name} ({db})"
                    f"{f' after {timeout}s' if _is_timeout_like_error(exc) else ''}: "
                    f"{exc.__class__.__name__}"
                    f"{f' status={status}' if status else ''}; "
                    f"action={action_content}"
                )
                raise
            backoff = min(_RETRY_MAX_BACKOFF_S, _RETRY_BACKOFF_S * (2**attempt))
            logger.warning(
                f"KB API retry {attempt + 1}/{_MAX_ATTEMPTS - 1} for {name} "
                f"({db}): {exc.__class__.__name__}"
                f"{f' status={status}' if status else ''}; sleep {backoff:.2f}s; "
                f"action={action_content}"
            )
            time.sleep(backoff)
    raise last_exc


@cache
def get_actions_api(db):
    """Check the service and return tool functions bound to the requested knowledge graph."""
    _log_base_url_once()

    url = _base_url(db)
    mcp_version = None
    try:
        # On first use, check the legacy endpoint or complete the MCP handshake.
        if _INTERACTIVE_API_BASE[db] or _GATEWAY:
            health_path = "test" if _INTERACTIVE_API_BASE[db] else "health"
            _HTTP_CLIENT.get(f"{url}/{health_path}", timeout=5.0).raise_for_status()
        else:
            result = _mcp_request(
                url,
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "fast-mcts-kbqa", "version": "1"},
                },
                timeout=5.0,
            )
            mcp_version = result["protocolVersion"]
            _mcp_request(
                url, "notifications/initialized", version=mcp_version, timeout=5.0
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Connection to {db} failed: {url}: {exc}") from exc

    def SearchNodes(query="", n_results=10, omit_length=500):
        """Build entity search arguments and call the tool."""
        args = {"query": query, "n_results": int(n_results)}
        if omit_length is not None:
            args["omit_length"] = int(omit_length)
        return _call_api_tool(db, "SearchNodes", args, mcp_version=mcp_version)

    if db == "fb":

        def SearchTypes(query="", n_results=10, omit_length=None):
            """Search Freebase types. The legacy gateway accepts an optional length limit."""
            if _INTERACTIVE_API_BASE[db]:
                return "Error: SearchTypes requires a compatible Freebase gateway."
            args = {"query": query, "n_results": int(n_results)}
            if omit_length is not None and mcp_version is None:
                args["omit_length"] = int(omit_length)
            return _call_api_tool(db, "SearchTypes", args, mcp_version=mcp_version)

    else:

        def SearchTypes(query="", n_results=10, omit_length=None):
            """Report that this knowledge graph does not support type search."""
            return "No support for SearchTypes"

    def SearchGraphPatterns(sparql="", semantic="", topN_return=10):
        """Find adjacent relations and optionally rank them by a text description."""
        sparql = (sparql or "").strip()
        if not sparql:
            return None
        args = {
            "sparql": sparql,
            "semantic": (semantic or None),
            "topN_vec": 400,
            "topN_return": int(topN_return),
        }
        return _call_api_tool(
            db,
            "SearchGraphPatterns",
            args,
            timeout=_SEARCH_GRAPH_PATTERNS_TIMEOUT_S,
            mcp_version=mcp_version,
        )

    def ExecuteSPARQL(sparql="", str_mode=True):
        """Execute a SPARQL query and return the results."""
        sparql = (sparql or "").strip()
        if not sparql:
            return None
        args = {
            "sparql": sparql,
            "str_mode": bool(str_mode),
        }
        return _call_api_tool(
            db,
            "ExecuteSPARQL",
            args,
            timeout=_DEFAULT_TOOL_TIMEOUT_S,
            mcp_version=mcp_version,
        )

    return SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL


def test_fb():
    SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api("fb")

    print(SearchNodes("millennium anxiety"))
    print(SearchTypes("computer game", n_results=5))

    s = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e ns:type.object.name "Germany"@en . }',
        semantic="official language",
    )
    print(s)

    s = ExecuteSPARQL(
        'SELECT DISTINCT ?x WHERE { ?e ns:type.object.name "Donald Trump"@en . ?e ns:people.person.place_of_birth ?x . }'
    )
    print(s)


def test_kqapro():
    """Check entity search and graph pattern tools for KQA Pro."""
    SearchNodes, _, SearchGraphPatterns, _ = get_actions_api("kqapro")
    s = SearchNodes("Kylie Minogue")
    print(s)

    s = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e <pred:name> "Kylie Minogue" . }',
        semantic="brother",
    )
    print(s)


if __name__ == "__main__":
    # python -m kb.client
    test_fb()
    print("-" * 100)
    test_kqapro()
