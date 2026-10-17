from pathlib import Path

import pytest

from scripts.verify_native_runtime import verify_junit


@pytest.mark.parametrize('body', ['', '<testcase><skipped/></testcase>', '<testcase><failure/></testcase>',
                                 '<testcase><error/></testcase>'])
def test_native_ci_rejects_empty_failed_and_skipped_selection(tmp_path, body):
    path = tmp_path / 'results.xml'
    path.write_text('<testsuites><testsuite>' + body + '</testsuite></testsuites>')
    with pytest.raises(ValueError):
        verify_junit(path)


def test_native_ci_accepts_only_executed_successful_tests(tmp_path):
    path = tmp_path / 'results.xml'
    path.write_text('<testsuites><testsuite><testcase/><testcase/></testsuite></testsuites>')
    assert verify_junit(path) == {'executed_tests': 2, 'skipped': 0}
