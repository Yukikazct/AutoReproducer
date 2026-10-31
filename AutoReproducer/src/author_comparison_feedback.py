"""Source navigation for rejected context quotes, never quotation repair."""
from copy import deepcopy
from difflib import SequenceMatcher
import re

from src.evidence_quotes import original_quote_spans


def context_quote_hints(attempt, excerpts):
    """Return bounded, single-excerpt slices from the already supplied view."""
    proposal = attempt.get('proposal') or {}
    if not isinstance(proposal, dict):
        return []
    rows = proposal.get('comparisons', [])
    if not isinstance(rows, list):
        return []
    citations = proposal.get('citations')
    declarations = {item['id']: item for item in (citations if isinstance(citations, list) else [])
                    if isinstance(item, dict) and isinstance(item.get('id'), str)}
    hints = []
    for error in attempt.get('validation_errors', []):
        match = re.match(r'^(reference|actual_context)\.context_quotes\.(\w+) must occur ',
                         str(error.get('error', '')))
        index = error.get('index')
        if not match or type(index) is not int or not 0 <= index < len(rows):
            continue
        side, field = match.groups()
        node = rows[index].get(side) if isinstance(rows[index], dict) else None
        if not isinstance(node, dict) or not isinstance(node.get('context_quotes'), dict):
            continue
        proposed = node['context_quotes'].get(field)
        if not isinstance(proposed, str) or not 4 <= len(proposed) <= 1200:
            continue
        terms = set(re.findall(r'\w{4,}', proposed.casefold()))
        proposed_words = re.findall(r'\w+', proposed.casefold())
        identifiers = node.get('citations')
        selected = set()
        for key in identifiers if isinstance(identifiers, list) else []:
            if isinstance(key, str):
                identifier = declarations.get(key, {}).get('evidence_id', key)
                if isinstance(identifier, str):
                    selected.add(identifier)
        source = 'pdf' if side == 'reference' else 'repository'
        candidates = []
        for order, (key, excerpt) in enumerate(excerpts.items()):
            if excerpt.get('source') != source:
                continue
            quote = excerpt['quote']
            words = list(re.finditer(r'\w+', quote))
            tokens = [word.group().casefold() for word in words]
            overlap = terms.intersection(tokens)
            if not overlap:
                continue
            # Similarity chooses a reading location, never validates a quote.
            match = SequenceMatcher(None, proposed_words, tokens, autojunk=False).find_longest_match()
            start = max(0, words[match.b].start() - 100)
            literal = bool(original_quote_spans(proposed, [(quote, key)], pdf_wrapping=source == 'pdf'))
            candidates.append(((-int(literal), -int(key in selected), -match.size, -len(overlap), order), {
                'actual_id': error.get('actual_id'), 'field': side + '.context_quotes.' + field,
                'evidence_id': key, 'source': source, 'already_selected': key in selected,
                **{locator: excerpt[locator] for locator in ('page', 'path') if locator in excerpt},
                'quote': quote[start:start + 600]}))
        hints.extend(item for _, item in sorted(candidates)[:3])
        if len(hints) >= 6:
            break
    return hints[:6]


def correction_source_view(attempt, view):
    """Focus a quote correction on whole provided sources; fall back if unsure."""
    if not attempt.get('context_quote_hints'):
        return view
    proposal = attempt.get('proposal')
    if not isinstance(proposal, dict) or not isinstance(proposal.get('comparisons'), list):
        return view
    declarations = proposal.get('citations')
    if not isinstance(declarations, list):
        return view
    aliases = {item['id']: item for item in declarations
               if isinstance(item, dict) and isinstance(item.get('id'), str)}
    excerpts, selected = view['evidence_excerpts'], set()
    for error in attempt.get('validation_errors', []):
        index = error.get('index')
        rows = proposal['comparisons']
        if type(index) is not int or not 0 <= index < len(rows) or not isinstance(rows[index], dict):
            return view
        for side in ('reference', 'actual_context', 'settings'):
            node = rows[index].get(side)
            if not isinstance(node, dict) or not isinstance(node.get('citations'), list):
                return view
            for key in node['citations']:
                if not isinstance(key, str):
                    return view
                declaration = aliases.get(key, {})
                identifier = declaration.get('evidence_id', key)
                if isinstance(identifier, str) and identifier in excerpts:
                    selected.add(identifier)
                else:
                    # Legacy literal citations are ambiguous to project without
                    # repeating validation; retain the full original view.
                    return view
    for hint in [*attempt['context_quote_hints'], *attempt.get('label_evidence_hints', [])]:
        identifier = hint.get('evidence_id') if isinstance(hint, dict) else None
        if not isinstance(identifier, str) or identifier not in excerpts:
            return view
        selected.add(identifier)
    if not selected:
        return view
    result = deepcopy(view)
    result['evidence_excerpts'] = {key: value for key, value in result['evidence_excerpts'].items() if key in selected}
    kept = list(result['evidence_excerpts'].values())
    pages = {item['page'] for item in kept if item['source'] == 'pdf'}
    paths = {item['path'] for item in kept if 'path' in item}
    result['pdf']['pages'] = [page for page in result['pdf']['pages'] if page['page'] in pages]
    result['repository']['files'] = {path: entry for path, entry in result['repository']['files'].items() if path in paths}
    result['file_inventory'] = {path: entry for path, entry in result['file_inventory'].items() if path in paths}
    repository = [item for item in kept if item['source'] == 'repository']
    result['source_selection'].update(scope='context_quote_correction',
        full_pdf_retained=all(key in selected for key, item in excerpts.items() if item['source'] == 'pdf'),
        repository_quote_characters=sum(len(item['quote']) for item in repository),
        repository_excerpt_count=len(repository), selected_repository_files=len(result['repository']['files']),
        selected_evidence_ids=list(result['evidence_excerpts']),
        omitted_sources_are_not_negative_evidence=True)
    return result
