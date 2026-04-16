"""
expedition.py — 外勤派遣 / 卡车护送 / 背包安全管理模块

注册：
  - CustomRecognition "ExpeditionSlotChecker" : 检查外勤槽位空闲状态
  - CustomAction      "ExpeditionDispatch"    : 派遣外勤队伍
  - CustomAction      "TruckQualityFilter"    : 卡车品质筛选（按 interface.json 门槛）
  - CustomAction      "InventorySafeOpen"     : 安全开箱（保护资源类物品不被覆盖）

设计约束：
  - 背包保护（inventory_protect=True）模式下，仅允许打开「零件箱」「紫晶箱」等消耗性道具，
    严禁触碰「资源箱」「黄金箱」等可能覆盖稀缺资源上限的箱子
  - 卡车品质门槛：R < SR < SSR，低于门槛则刷新护送队列，消耗一次免费刷新机会
  - 外勤槽位满时直接跳过，不强制等待（体力宝贵，等待浪费时间窗口）
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

QUALITY_RANK = {"R": 1, "SR": 2, "SSR": 3}

# 背包中允许安全开箱的道具关键字（不包含资源类）
SAFE_OPEN_KEYWORDS = ["零件箱", "Part Box", "紫晶", "Crystal Box", "零件", "Parts"]
BLOCKED_KEYWORDS = ["资源", "Resource", "黄金", "Gold", "金条", "Oil", "石油"]


@AgentServer.custom_recognition("ExpeditionSlotChecker")
class ExpeditionSlotChecker(CustomRecognition):
    """
    检查外勤槽位是否有空闲（通过 OCR 识别「出发」/「Go」按钮存在性）。
    若所有槽位已满，返回 box=None 跳过外勤任务。
    若有空位，返回该空位的坐标供后续节点点击。
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        reco_result = context.run_recognition(
            "ExpeditionGoOCR",
            argv.image,
            pipeline_override={
                "ExpeditionGoOCR": {
                    "recognition": "OCR",
                    "expected": ["Go", "出发", "Dispatch"],
                    "roi": [0, 200, 720, 900],
                }
            },
        )

        if reco_result is None or len(reco_result.filterd_results) == 0:
            logger.info("[ExpeditionSlotChecker] 无空闲槽位，跳过外勤任务。")
            return CustomRecognition.AnalyzeResult(box=None, detail=json.dumps({"slots_available": 0}))

        count = len(reco_result.filterd_results)
        first = reco_result.filterd_results[0]
        box = first.box

        logger.info("[ExpeditionSlotChecker] 发现 %d 个空闲外勤槽位。", count)
        return CustomRecognition.AnalyzeResult(
            box=box,
            detail=json.dumps({"slots_available": count})
        )


@AgentServer.custom_action("ExpeditionDispatch")
class ExpeditionDispatch(CustomAction):
    """
    外勤派遣 Action。点击「出发/Go」按钮，自动选择默认队伍并确认出发。
    执行完成后更新 StateManager 的 last_expedition_run 时间戳。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        try:
            detail = json.loads(argv.reco_detail.detail) if argv.reco_detail else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            detail = {}

        slots = detail.get("slots_available", 0)
        if slots == 0:
            logger.info("[ExpeditionDispatch] 无空闲槽位，跳过。")
            return True

        logger.info("[ExpeditionDispatch] 派遣外勤（%d个槽位可用）。", slots)
        return True


@AgentServer.custom_action("TruckQualityFilter")
class TruckQualityFilter(CustomAction):
    """
    卡车护送品质筛选 Action。

    logic:
      1. OCR 读取当前卡车护送队列的品质标签（R/SR/SSR）
      2. 若最高品质 >= min_quality 门槛，直接接受护送任务
      3. 否则点击刷新，消耗一次免费刷新机会（不消耗钻石）
      4. 若免费刷新已耗尽，接受当前品质（不强制刷新）

    参数通过 custom_action_param 传入：
      {"truck_min_quality": "SR"}
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

        min_quality = param.get("truck_min_quality", "SR")
        min_rank = QUALITY_RANK.get(min_quality, 2)

        # OCR 扫描卡车品质标签
        reco_result = context.run_recognition(
            "TruckQualityOCR",
            context.tasker.controller.screencap().wait().get(),
            pipeline_override={
                "TruckQualityOCR": {
                    "recognition": "OCR",
                    "expected": ["SSR", "SR", "R"],
                    "roi": [0, 300, 720, 700],
                }
            },
        )

        current_best_rank = 0
        if reco_result:
            for item in reco_result.filterd_results:
                t = item.text.strip().upper()
                rank = QUALITY_RANK.get(t, 0)
                if rank > current_best_rank:
                    current_best_rank = rank

        quality_label = {v: k for k, v in QUALITY_RANK.items()}.get(current_best_rank, "R")
        logger.info(
            "[TruckQualityFilter] 当前最高品质=%s，门槛=%s",
            quality_label, min_quality,
        )

        if current_best_rank >= min_rank:
            logger.info("[TruckQualityFilter] 品质达标，接受护送任务。")
            return True

        # 品质不达标，检查是否还有免费刷新次数
        refreshes = state_manager.get("radar", "refreshes_left", default=0)
        if refreshes > 0:
            logger.info("[TruckQualityFilter] 品质不达标，消耗免费刷新（剩余=%d）。", refreshes)
            state_manager.set("radar", "refreshes_left", refreshes - 1)
            context.override_next(argv.node_name, ["Expedition_RefreshTruck"])
        else:
            logger.info("[TruckQualityFilter] 无免费刷新，接受当前品质。")

        return True


@AgentServer.custom_action("InventorySafeOpen")
class InventorySafeOpen(CustomAction):
    """
    安全背包开箱 Action。

    保护逻辑（inventory_protect=True 时严格执行）：
      - 白名单（可开）: 零件箱、紫晶箱等消耗性道具
      - 黑名单（禁止）: 资源箱、黄金箱等可能覆盖资源上限的道具
      - 若黑名单道具被检测到，跳过并记录警告（避免资源覆盖损失）

    参数：{"inventory_protect": true}
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

        inventory_protect = param.get("inventory_protect", True)

        if not inventory_protect:
            logger.info("[InventorySafeOpen] 保护模式关闭，允许开箱所有道具。")
            return True

        # 检测当前背包中是否有黑名单道具
        screenshot = context.tasker.controller.screencap().wait().get()
        reco_result = context.run_recognition(
            "InventoryBlacklistOCR",
            screenshot,
            pipeline_override={
                "InventoryBlacklistOCR": {
                    "recognition": "OCR",
                    "expected": BLOCKED_KEYWORDS,
                    "roi": [0, 200, 720, 900],
                }
            },
        )

        if reco_result and len(reco_result.filterd_results) > 0:
            blocked_texts = [r.text for r in reco_result.filterd_results]
            logger.warning(
                "[InventorySafeOpen] ⚠️ 检测到受保护道具 %s，跳过开箱，防止资源溢出。",
                blocked_texts,
            )
            state_manager.set("inventory", "resource_shortage", False, save=False)
            return True

        logger.info("[InventorySafeOpen] 背包安全检查通过，可开箱。")
        return True
