"""Report real command completion without paper or model acceptance gates."""


def completed_execution(execution, *, mock_mode=False):
    final = execution.get('final') or {}
    if (execution.get('success') is False or final.get('success') is False
            or final.get('exit_code') not in (None, 0) or execution.get('not_runnable')):
        raise RuntimeError(execution.get('reason') or final.get('stderr') or '代码执行失败。')
    if mock_mode:
        return {'status': 'mock_demo', 'result_level': 'mock_demo', 'is_reproduced': None,
                'reviews_removed': True, 'paper_metrics_verified': False,
                'reason': '模拟流程结束；未执行真实作者代码。'}
    steps = execution.get('steps') or [final]
    if (execution.get('executed', final.get('exit_code') is not None) is not True
            or execution.get('success', final.get('success')) is not True
            or execution.get('cancelled') or execution.get('interrupted')
            or any(step.get('exit_code') != 0 or step.get('timed_out')
                   or step.get('cancelled') or step.get('interrupted')
                   or step.get('success') is False for step in steps)):
        raise RuntimeError(execution.get('reason') or final.get('stderr')
                           or '作者命令尚未正常结束，不能标记运行完成。')
    return {'status': 'author_code_completed', 'result_level': 'author_code_completed',
            'execution_status': 'author_code_completed', 'is_reproduced': None,
            'paper_metrics_verified': False, 'reviews_removed': True,
            'completion_checks': {'all_required_author_commands_returned': True,
                                  'owned_processes_finished': True},
            'reason': '作者命令已正常结束；未进行论文、协议或数值审核。'}
