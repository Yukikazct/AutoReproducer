"""Read public repository search results when the GitHub API is unavailable."""
from html import unescape
from html.parser import HTMLParser
import json
import re
import urllib.parse
import urllib.request


_MAX_BYTES = 2 * 1024 * 1024
_NAME = re.compile(r'[A-Za-z0-9_.-]{1,100}')


class _SearchData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.selected = False
        self.parts = []
        self.documents = []

    def handle_starttag(self, tag, attrs):
        self.selected = (tag == 'script' and dict(attrs).get('type') == 'application/json'
                         and dict(attrs).get('data-target') == 'react-app.embeddedData')
        if self.selected:
            self.parts = []

    def handle_data(self, data):
        if self.selected:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.selected:
            self.documents.append(''.join(self.parts))
            self.selected = False


def parse_repository_results(content, *, limit=5):
    """Accept repository identities only from the site's structured results."""
    if not isinstance(content, str) or len(content) > _MAX_BYTES:
        return []
    parser = _SearchData()
    parser.feed(content)
    rows, seen = [], set()
    for document in parser.documents:
        try:
            payload = json.loads(document).get('payload', {})
            if not isinstance(payload, dict):
                continue
            route = payload.get('blackbirdSearchRoute', payload)
            items = route.get('results', []) if isinstance(route, dict) else []
        except (ValueError, AttributeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items[:100]:
            if not isinstance(item, dict) or item.get('public') is False:
                continue
            repo = item.get('repo') or {}
            repo = repo.get('repository') if isinstance(repo, dict) else None
            if not isinstance(repo, dict):
                continue
            owner, name = repo.get('owner_login'), repo.get('name')
            if (not isinstance(owner, str) or not isinstance(name, str)
                    or not _NAME.fullmatch(owner) or not _NAME.fullmatch(name)
                    or owner in {'.', '..'} or name in {'.', '..'}):
                continue
            full_name = owner + '/' + name
            if full_name.casefold() in seen:
                continue
            seen.add(full_name.casefold())
            description = item.get('hl_trunc_description')
            description = unescape(re.sub(r'<[^>]*>', '', description))[:2000] if isinstance(description, str) else ''
            stars = item.get('followers')
            rows.append({'full_name': full_name, 'name': name, 'description': description,
                         'stargazers_count': stars if type(stars) is int and stars >= 0 else 0})
            if len(rows) >= limit:
                return rows
    return rows


def search_repository_page(query, *, limit=5, timeout=8.0):
    url = 'https://github.com/search?' + urllib.parse.urlencode({'q': query, 'type': 'repositories'})
    request = urllib.request.Request(url, headers={
        'User-Agent': 'AutoReproducer/1.0', 'Accept': 'text/html'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            return [], url, 'Public repository search exceeded the response size limit'
        return parse_repository_results(raw.decode('utf-8', errors='replace'), limit=limit), url, ''
    except (OSError, ValueError) as exc:
        return [], url, type(exc).__name__ + ': public repository search unavailable'
