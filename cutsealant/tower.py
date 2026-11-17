"""Tower functionality extracted from CutSealant V6.0.7."""


class TowerController:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context
        self.last_status = None

    def _call_light_once(self, status):
        """Call light control only when status actually changes"""
        ctx = self.context
        if ctx.last_light_status != status:
            print(f"Light status change: {ctx.last_light_status} → {status}")
            result = ctx._call_light(status)
            if result == status:  # 成功才更新狀態
                ctx.last_light_status = status
                return status
            else:
                print(f"⚠️ Light change failed, retry next cycle")
                return ctx.last_light_status  # 返回舊狀態，下次會重試
        return status

    def _call_light(self, status):
        """Control stack light status"""
        ctx = self.context
        commands = {
            'Standby': ['2', '0', '1', '0', '0', '0'],
            'Tracking': ['2', '0', '2', '0', '0', '0'],
            'Pass': ['2', '0', '0', '1', '0', '0'],
            'NG': ['2', '1', '0', '0', '0', '0']
        }

        mode = ctx.light_tower_config.get('mode', 'usb')
        if mode == 'mock':
            print(f'[Mock light tower] Status: {status}')
            if ctx.logger:
                ctx.logger.log_system_event("status_change", {"new_status": status})
            return status

        try:
            if status in commands:
                result = ctx.command_runner(['python', ctx.paths['stack_light_path']] + commands[status])
                if result != 0:
                    print(f"⚠️ Light command failed with code: {result}")
                    return None
                if status == 'NG':
                    ctx.command_runner(['python', ctx.paths['stack_light_path']] + ['4', '3', '1', '1', '14'])
        except Exception as e:
            error_msg = f"Light control error: {e}"
            print(error_msg)
            if ctx.logger:
                ctx.logger.log_error("light_control_error", e, {"status": status})
            return None

        print(f'Status: {status}')
        if ctx.logger:
            ctx.logger.log_system_event("status_change", {"new_status": status})
        return status
