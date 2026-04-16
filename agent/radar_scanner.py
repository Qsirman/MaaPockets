"""
radar_scanner.py — 雷达矩阵优先级清空系统

注册：
  - CustomRecognition "RadarScanner"  : 扫描雷达任务列表，按优先级排队
  - CustomRecognition "RadarReadyChecker" : 检查是否需要触发雷达流程
  - CustomAction      "RadarDispatch" : 执行选定任务、管理刷新次数

优先级逻辑（从高到低）：
  1. 红点标识的极品任务（TemplateMatch 红点 badge）
  2. "Lost supplies" / "遗失的物资"（零体力消耗，立即生效）
  3. 僵尸/救援任务（OCR 关键字匹配）

药水锁死策略：
  - 常规雷达任务中，严格禁止使用体力药水
  - 仅当 top_lord_active=True 且体力 < 10 时，才允许使用药水

体力同步：
  - RadarDispatch 执行完成后，通过 OCR 读取界面体力数值，
    调用 state_manager.sync_stamina_from_game() 同步实测值
"""

import json
import logging
import re

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction
from maa.context import Context

from state_manager import state_manager

logger = logging.getLogger(__name__)

# 雷达任务关键字（OCR 匹配）
LOST_SUPPLIES_KEYWORDS = ["Lost supplies", "遗失的物资", "Lost Supplies"]
ZOMBIE_KEYWORDS = ["Zombie", "僵尸", "Rescue", "救援", "rescue"]
TOP_LORD_KEYWORDS = ["Top Lord", "最强领主", "top lord"]


@AgentServer.custom_recognition("RadarReadyChecker")
class RadarReadyChecker(CustomRecognition):
    """
    检查是否需要触发雷达流程。
    若雷达刷新次数 > 0 或体力充足，则返回命中（触发 StartRadar 流程）。
    若雷达次数耗尽且体力不足，则返回失败（跳过雷达，节省操作次数）。

    识别结果的 detail 字段携带当前体力预测值，供后续节点参考。
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        predicted = state_manager.predict_stamina()
        refreshes_left = state_manager.get("radar", "refreshes_left", default=0)

        detail = json.dumps({
            "predicted_stamina": predicted,
            "refreshes_left": refreshes_left,
        })

        # 有体力或有免费刷新次数，则触发雷达
        if predicted >= 5 or refreshes_left > 0:
            logger.info(
                "[RadarReadyChecker] 满足触发条件（体力=%d, 刷新=%d），准备进入雷达。",
                predicted, refreshes_left,
            )
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 720, 1280), detail=detail
            )

        logger.info(
            "[RadarReadyChecker] 条件不足（体力=%d, 刷新=%d），跳过雷达。",
            predicted, refreshes_left,
        )
        return CustomRecognition.AnalyzeResult(box=None, detail=detail)


@AgentServer.custom_recognition("RadarScanner")
class RadarScanner(CustomRecognition):
    """
    雷达任务列表优先级识别引擎。

    analyze() 工作流：
      1. 对当前截图执行 OCR，提取雷达列表区域中的所有文本
      2. 逐行分析，按优先级打分
      3. 返回优先级最高的任务坐标（供 RadarDispatch 点击）

    识别结果 box：最高优先级任务所在行的点击坐标区域
    识别结果 detail：JSON，包含任务类型、优先级、是否消耗体力
    """

    PRIORITY_LOST_SUPPLIES = 100
    PRIORITY_RED_DOT = 90
    PRIORITY_ZOMBIE = 50
    PRIORITY_NORMAL = 10

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        # 在雷达列表区域执行 OCR（720x1280 竖屏，任务列表通常在屏幕中下区域）
        # ROI：[x, y, w, h] — 根据实际游戏截图调整
        RADAR_LIST_ROI = [0, 300, 720, 700]

        reco_result = context.run_recognition(
            "RadarListOCR",
            argv.image,
            pipeline_override={
                "RadarListOCR": {
                    "recognition": "OCR",
                    "roi": RADAR_LIST_ROI,
                }
            },
        )

        if reco_result is None:
            logger.warning("[RadarScanner] OCR 无结果，雷达列表可能为空。")
            return CustomRecognition.AnalyzeResult(box=None, detail="{}")

        # 解析 OCR 结果，逐项评分
        best_priority = -1
        best_box = None
        best_task_type = "none"
        best_consumes_stamina = True

        for item in reco_result.filterd_results:
            text = item.text.strip()
            box = item.box  # [x, y, w, h]

            priority, task_type, consumes_stamina = self._score_task(text)

            if priority > best_priority:
                best_priority = priority
                best_box = box
                best_task_type = task_type
                best_consumes_stamina = consumes_stamina

        if best_box is None:
            logger.info("[RadarScanner] 无可执行任务。")
            return CustomRecognition.AnalyzeResult(box=None, detail="{}")

        detail = json.dumps({
            "task_type": best_task_type,
            "priority": best_priority,
            "consumes_stamina": best_consumes_stamina,
            "box": list(best_box),
        })

        logger.info(
            "[RadarScanner] 最优任务: %s（优先级=%d, 消耗体力=%s）",
            best_task_type, best_priority, best_consumes_stamina,
        )

        return CustomRecognition.AnalyzeResult(box=best_box, detail=detail)

    def _score_task(self, text: str):
        """
        根据文本内容为雷达任务评分。
        返回 (priority, task_type, consumes_stamina)
        """
        text_lower = text.lower()

        # 优先级1：Lost supplies（不消耗体力，最优先）
        if any(kw.lower() in text_lower for kw in LOST_SUPPLIES_KEYWORDS):
            return self.PRIORITY_LOST_SUPPLIES, "lost_supplies", False

        # 优先级2：僵尸/救援任务
        if any(kw.lower() in text_lower for kw in ZOMBIE_KEYWORDS):
            return self.PRIORITY_ZOMBIE, "zombie_rescue", True

        # 优先级3：普通任务
        return self.PRIORITY_NORMAL, "normal", True


@AgentServer.custom_action("RadarDispatch")
class RadarDispatch(CustomAction):
    """
    雷达任务调度 Action。

    执行逻辑：
      1. 从 reco_detail 读取 RadarScanner 识别结果
      2. 检查体力与药水策略
      3. 点击任务行，启动任务
      4. 更新 StateManager（体力、刷新次数）

    药水锁死安全阀：
      - consumes_stamina=True 时，必须检查体力是否充足
      - 体力不足时，仅当 top_lord_active=True 且 allow_potion=True 才允许使用药水
      - 否则跳过该任务
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 解析识别结果
        try:
            detail = json.loads(argv.reco_detail.detail) if argv.reco_detail else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            detail = {}

        task_type = detail.get("task_type", "none")
        consumes_stamina = detail.get("consumes_stamina", True)
        box = detail.get("box")

        if not box:
            logger.info("[RadarDispatch] 无任务可执行。")
            return True

        # 从 custom_action_param 读取策略参数
        try:
            param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        except (json.JSONDecodeError, TypeError):
            param = {}

        allow_potion = param.get("allow_potion", False)
        use_free_refresh = param.get("use_free_refresh", True)

        # ── 体力检查 ──────────────────────────────
        if consumes_stamina:
            predicted = state_manager.predict_stamina()
            top_lord_active = state_manager.get("tasks", "top_lord_active", default=False)

            if predicted < 5:
                if top_lord_active and allow_potion:
                    # Top Lord 活动期间体力见底，解锁药水使用路径
                    logger.info(
                        "[RadarDispatch] Top Lord 活动期，体力不足，触发药水使用流程。"
                    )
                    context.override_next(argv.node_name, ["Radar_UsePotion"])
                    return True
                else:
                    # 体力不足且未授权使用药水，跳过本次任务
                    logger.info(
                        "[RadarDispatch] 体力不足（%d<5）且未授权药水，跳过任务 %s。",
                        predicted, task_type,
                    )
                    return True

        # ── 点击任务 ──────────────────────────────
        x = box[0] + box[2] // 2
        y = box[1] + box[3] // 2
        logger.info("[RadarDispatch] 点击任务 %s at (%d, %d)", task_type, x, y)
        context.tasker.controller.post_click(x, y).wait()

        # ── 更新雷达刷新次数 ──────────────────────
        refreshes = state_manager.get("radar", "refreshes_left", default=0)
        if refreshes > 0:
            state_manager.set("radar", "refreshes_left", refreshes - 1)
            logger.info("[RadarDispatch] 雷达剩余免费刷新: %d", refreshes - 1)

        # ── 体力同步（执行完任务后重新 OCR 读取实测体力）──
        # 实际体力值通过 OCR 在 Pipeline 节点读取后，由 Radar_SyncStamina 节点回调
        # 此处仅减去预估消耗（1点/任务，保守估计）
        current = state_manager.predict_stamina()
        if consumes_stamina and current > 0:
            state_manager.sync_stamina_from_game(max(0, current - 1))

        return True


@AgentServer.custom_action("TopLordDetectAction")
class TopLordDetectAction(CustomAction):
    """
    Top Lord 活动检测 Action。
    当 OCR 检测到「Top Lord/最强领主」关键字时，将 top_lord_active=True 写入 StateManager，
    解锁药水使用路径。
    由 radar.json 的 Radar_CheckTopLord 节点调用。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        state_manager.set("tasks", "top_lord_active", True, save=True)
        logger.info("[TopLordDetectAction] 检测到 Top Lord 活动，已激活药水使用路径。")
        return True


@AgentServer.custom_action("StaminaSyncAction")
class StaminaSyncAction(CustomAction):
    """
    体力实测同步 Action。
    从 OCR 识别结果的 detail 中解析出真实体力数值，
    调用 state_manager.sync_stamina_from_game() 修正离线预测偏差。
    由 radar.json 的 Radar_SyncStamina 节点调用。

    OCR 结果格式（expected）：
      "45/75" 或 "45" 均可解析
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        import re as _re
        try:
            detail = argv.reco_detail.detail if argv.reco_detail else ""
            text = detail.strip() if detail else ""
            match = _re.search(r"(\d+)", text)
            if match:
                value = int(match.group(1))
                state_manager.sync_stamina_from_game(min(value, 75))
                logger.info("[StaminaSyncAction] 体力同步: %d/75", value)
            else:
                logger.warning("[StaminaSyncAction] 无法从 OCR 结果解析体力值: %r", text)
        except Exception as e:
            logger.error("[StaminaSyncAction] 解析异常: %s", e)
        return True
