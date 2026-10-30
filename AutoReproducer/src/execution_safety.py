"""Read execution stop evidence consistently without importing runtime services."""
import re


_WRAPPERS = ('execution', 'final', 'effective_env_config', 'environment',
             'steps', 'attempts', 'stages')
_RECEIPTS = ('cleanup', 'process_cleanup', 'cleanup_receipt', 'environment_cleanup')


def iter_execution_records(execution):
    """Follow known wrappers; preserve receipt meaning for shared dictionaries."""
    pending, seen = [(execution, False)], set()
    while pending:
        record, receipt = pending.pop()
        if not isinstance(record, (dict, list, tuple)):
            continue
        identity = (id(record), receipt)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(record, (list, tuple)):
            pending.extend((item, receipt) for item in record)
            continue
        yield record, receipt
        pending.extend((record.get(key), True) for key in _RECEIPTS)
        pending.extend((record.get(key), False) for key in _WRAPPERS)


def cleanup_is_unconfirmed(execution):
    """Any explicit uncertainty stays blocking, including nested attempts."""
    diagnostics = []
    for record, receipt in iter_execution_records(execution):
        if (record.get('cleanup_confirmed') is False or record.get('process_cleanup_confirmed') is False
                or any(record.get(key) for key in ('cleanup_failed', 'cleanup_unconfirmed',
                                                  'cleanup_error', 'cleanup_errors', 'processes_alive'))
                or record.get('state') == 'cleanup_unconfirmed'
                or record.get('status') in ('cleanup_failed', 'cleanup_unconfirmed')
                or receipt and (record.get('confirmed') is False
                                or record.get('status') in ('failed', 'unconfirmed'))):
            return True
        diagnostics.extend(str(record.get(key) or '') for key in ('reason', 'stderr', 'stdout'))
    return bool(re.search(r'process[- ](?:tree|group) cleanup (?:failed|was denied)'
                          r'|\bcleanup_unconfirmed\b', '\n'.join(diagnostics), re.I))
