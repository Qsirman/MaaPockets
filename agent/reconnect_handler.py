"""
reconnect_handler.py — 断线重连 CustomAction 回调

注册两个 AgentServer CustomAction：
  - ReconnectAction        : 指数退避等待 + 记录失败次数
  - ReconnectSuccessAction : 清零失败计数，记录重连成功

这两个 Action 被 global_guard.json 中的 Guard_Backoff_* 节点调用，
是 Pipeline 层与 Python 层状态机之间的唯一桥梁。
"""

import time
import logging
import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from state_manager import state_manager

logger = logging.getLogger(__name__)


@AgentServer.custom_action("ReconnectAction")
class ReconnectAction(CustomAction):
    """
    指数退避重连等待 Action。

    Pipeline 节点传入 custom_action_param.backoff_level 控制等待时长：
      level 1 → 1s, 2 → 2s, 3 → 4s, 4 → 8s, 5 → 16s,
      6 → 32s, 7 → 64s, 8 → 256s（上限）

    每次调用递增 StateManager 中的 fail_count，
    供后续调度逻辑判断是否需要降低运营强度。

    安全防护逻辑：
      - 等待期间 sleep 分段执行（每秒检查一次），保证系统响应性
      - 若 fail_count > 8，主动发出警告日志，提示人工介入可能需要
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 解析 backoff_level 参数
        try:
            param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        except (json.JSONDecodeError, TypeError):
            param = {}

        backoff_level = int(param.get("backoff_level", 1))
        wait_sec = min(2 ** (backoff_level - 1), 256)

        logger.warning(
            "[ReconnectAction] 重连等待 %ds（退避等级=%d）",
            wait_sec, backoff_level,
        )

        # 记录失败，获取当前指数退避时间（由 StateManager 维护）
        state_manager.record_reconnect_attempt(success=False)

        fail_count = state_manager.get("reconnect", "fail_count", default=0)
        if fail_count > 8:
            logger.error(
                "[ReconnectAction] 连续失败 %d 次，建议人工检查网络或游戏状态！",
                fail_count,
            )

        # 分段等待，保证调度线程可以响应停机信号
        waited = 0
        while waited < wait_sec:
            time.sleep(1)
            waited += 1

        logger.info("[ReconnectAction] 等待结束，继续尝试重连。")
        return True


@AgentServer.custom_action("ReconnectSuccessAction")
class ReconnectSuccessAction(CustomAction):
    """
    重连成功 Action。
    清零 StateManager 中的失败计数，记录重连时间戳。
    由 global_guard.json 的 Guard_ReconnectSuccess 节点调用。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        state_manager.record_reconnect_attempt(success=True)
        logger.info("[ReconnectSuccessAction] 重连成功，状态已重置。")
        return True


@AgentServer.custom_action("StartAFKAction")
class StartAFKAction(CustomAction):
    """
    AFK全自动运营入口 Action。
    从 custom_action_param 中读取 intensity/allow_potion 等参数，
    更新到 StateManager，供 Scheduler 后续读取。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        try:
            param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        except (json.JSONDecodeError, TypeError):
            param = {}

        intensity = param.get("intensity", "标准")
        allow_potion = param.get("allow_potion", False)

        # 将界面配置写入 StateManager，供 Scheduler 读取
        state_manager.update_section("scheduler", {"intensity": intensity})
        if allow_potion:
            state_manager.set("tasks", "top_lord_active", True, save=True)

        logger.info("[StartAFKAction] 运营参数已更新：强度=%s, 药水=%s", intensity, allow_potion)
        return True


@AgentServer.custom_action("DailyMarkAction")
class DailyMarkAction(CustomAction):
    """
    每日任务完成标记 Action。
    根据 custom_action_param.task 字段更新 StateManager.tasks 对应字段：
      "donated"          → tasks.donated = True
      "researched"       → tasks.researched = True
      "island_battle_done" → tasks.island_battle_done = True
      "all_done"         → 以上三项全部 True（一次性原子落盘）
    由 daily.json 的各 Daily_Mark* 节点调用。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        try:
            param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        except (json.JSONDecodeError, TypeError):
            param = {}

        task = param.get("task", "")

        if task == "all_done":
            state_manager.update_section("tasks", {
                "donated": True,
                "researched": True,
                "island_battle_done": True,
            }, save=True)
            logger.info("[DailyMarkAction] 所有每日任务已标记完成。")
        elif task in ("donated", "researched", "island_battle_done"):
            state_manager.set("tasks", task, True, save=True)
            logger.info("[DailyMarkAction] 标记 tasks.%s = True", task)
        else:
            logger.warning("[DailyMarkAction] 未知 task 参数: %r", task)

        return True


@AgentServer.custom_recognition("DailyStatusChecker")
class DailyStatusChecker(CustomRecognition):
    """
    每日任务完成状态检查 CustomRecognition。
    读取 StateManager.tasks 中的 donated/researched/island_battle_done 标记。
    若任意一项未完成，则返回命中（触发每日任务流程）；全部完成则返回 None（跳过）。
    由 main.json 的 AFK_CheckDaily 节点调用。
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        donated = state_manager.get("tasks", "donated", default=False)
        researched = state_manager.get("tasks", "researched", default=False)
        island_done = state_manager.get("tasks", "island_battle_done", default=False)

        all_done = donated and researched and island_done
        detail = json.dumps({
            "donated": donated,
            "researched": researched,
            "island_battle_done": island_done,
        })

        if all_done:
            logger.info("[DailyStatusChecker] 所有每日任务已完成，跳过。")
            return CustomRecognition.AnalyzeResult(box=None, detail=detail)

        logger.info(
            "[DailyStatusChecker] 每日任务未完成（捐赠=%s, 科研=%s, 岛屿=%s），触发流程。",
            donated, researched, island_done,
        )
        return CustomRecognition.AnalyzeResult(box=(0, 0, 720, 1280), detail=detail)
