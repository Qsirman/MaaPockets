"""
merge_engine.py — 基建与兵团空间坐标映射合并引擎

注册：
  - CustomRecognition "MergeScanner"  : 扫描屏幕内所有悬浮等级数字，构建坐标集合
  - CustomAction      "MergeSwipe"    : 对同等级单位执行 post_swipe 拖拽合成
  - CustomAction      "BlueprintCheck": 检测图纸储备并触发预警

算法约束（来自设计文档）：
  1. analyze() 中接收 AnalyzeArg 图像矩阵
  2. OCR 提取屏幕内所有等级数字，构建 [(level, x, y), ...] 集合
  3. 遍历同类下找到两个数值相等的坐标对 (x1,y1) 和 (x2,y2)
  4. run() 阶段调用 ctx.controller.post_swipe(x1,y1,x2,y2) 执行拖拽合成

安全防护：
  - 等级 0 或非数字 OCR 结果直接忽略，防止误操作
  - 坐标距离过近（< 20px）时跳过，防止同一单位被识别为两个
  - 每次合成后等待画面稳定（pre/post_wait_freezes 由 Pipeline 控制）
"""

import json
import logging
import re
from typing import Optional

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction
from maa.context import Context

from state_manager import state_manager

logger = logging.getLogger(__name__)

# 合并扫描 ROI：兵团/基建区域（竖屏 720x1280，主视图区域）
MERGE_ROI = [0, 150, 720, 950]
# 最小坐标间距（防止同一单位被识别两次）
MIN_COORD_DISTANCE = 20


@AgentServer.custom_recognition("MergeScanner")
class MergeScanner(CustomRecognition):
    """
    等级数字空间坐标映射识别引擎。

    analyze() 工作流：
      1. 在 MERGE_ROI 区域执行 OCR，提取所有数字文本及其坐标
      2. 筛选出纯数字结果（等级标识），构建 [(level, cx, cy), ...] 列表
      3. 在列表中找到第一对 level 相同的两个坐标
      4. 返回这对坐标（打包进 detail JSON），供 MergeSwipe 使用

    detail 格式：
      {"found": true/false, "x1": int, "y1": int, "x2": int, "y2": int, "level": int}
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        # 使用内置 OCR 在合并区域扫描所有数字
        reco_result = context.run_recognition(
            "MergeAreaOCR",
            argv.image,
            pipeline_override={
                "MergeAreaOCR": {
                    "recognition": "OCR",
                    "roi": MERGE_ROI,
                }
            },
        )

        if reco_result is None:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=json.dumps({"found": False})
            )

        # 构建等级坐标列表
        level_coords: list[tuple[int, int, int]] = []  # (level, cx, cy)

        for item in reco_result.filterd_results:
            text = item.text.strip()
            # 只保留纯数字且在合理等级范围内（1-20）
            if re.fullmatch(r"\d{1,2}", text):
                level = int(text)
                if 1 <= level <= 20:
                    box = item.box  # [x, y, w, h]
                    cx = box[0] + box[2] // 2
                    cy = box[1] + box[3] // 2
                    level_coords.append((level, cx, cy))

        if len(level_coords) < 2:
            logger.debug("[MergeScanner] 可合并单位不足（找到 %d 个）", len(level_coords))
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=json.dumps({"found": False})
            )

        # 寻找第一对同等级且间距足够的坐标
        match = self._find_merge_pair(level_coords)
        if match is None:
            logger.info("[MergeScanner] 无同等级可合并对。")
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=json.dumps({"found": False})
            )

        level, x1, y1, x2, y2 = match
        logger.info(
            "[MergeScanner] 找到合并对：等级=%d, (%d,%d) → (%d,%d)",
            level, x1, y1, x2, y2,
        )

        detail = json.dumps({
            "found": True,
            "level": level,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
        })

        # 返回覆盖两个单位的 box（用于 Pipeline 的 action target）
        merged_box = (
            min(x1, x2) - 20,
            min(y1, y2) - 20,
            abs(x2 - x1) + 40,
            abs(y2 - y1) + 40,
        )
        return CustomRecognition.AnalyzeResult(box=merged_box, detail=detail)

    def _find_merge_pair(
        self,
        level_coords: list[tuple[int, int, int]],
    ) -> Optional[tuple[int, int, int, int, int]]:
        """
        遍历坐标列表，找到第一对等级相同且坐标距离 >= MIN_COORD_DISTANCE 的单位对。
        返回 (level, x1, y1, x2, y2) 或 None。

        降维打击算法：
          O(n²) 遍历，但实际单位数量很少（通常 < 20），性能完全足够。
          找到第一对立即返回，不做全局最优搜索（贪心策略，符合游戏实时性要求）。
        """
        n = len(level_coords)
        for i in range(n):
            lv_i, x_i, y_i = level_coords[i]
            for j in range(i + 1, n):
                lv_j, x_j, y_j = level_coords[j]
                if lv_i != lv_j:
                    continue
                # 检查坐标距离，防止误识别同一单位
                dist = ((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5
                if dist < MIN_COORD_DISTANCE:
                    continue
                return (lv_i, x_i, y_i, x_j, y_j)
        return None


@AgentServer.custom_action("MergeSwipe")
class MergeSwipe(CustomAction):
    """
    执行拖拽合成操作。

    从 reco_detail 中读取 MergeScanner 提供的坐标对，
    调用 post_swipe(x1, y1, x2, y2, duration=500) 模拟手指拖拽。

    多进程解耦考量：
      post_swipe 是异步调用，必须 .wait() 等待执行完成，
      否则下一个 Pipeline 节点可能在拖拽尚未完成时截图，导致识别错误。
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

        if not detail.get("found", False):
            logger.info("[MergeSwipe] 无合并任务，跳过。")
            return True

        x1 = detail["x1"]
        y1 = detail["y1"]
        x2 = detail["x2"]
        y2 = detail["y2"]
        level = detail.get("level", "?")

        logger.info(
            "[MergeSwipe] 执行合并拖拽：等级=%s, (%d,%d) → (%d,%d)",
            level, x1, y1, x2, y2,
        )

        # post_swipe 参数：起点x, 起点y, 终点x, 终点y, 持续时间ms
        context.tasker.controller.post_swipe(x1, y1, x2, y2, 500).wait()

        logger.info("[MergeSwipe] 拖拽合成完成。")
        return True


@AgentServer.custom_action("BlueprintCheck")
class BlueprintCheck(CustomAction):
    """
    图纸储备预警 Action。
    当 OCR 检测到「图纸不足」关键字时，将 blueprint_warning=True 写入 StateManager，
    触发日志警告，提醒玩家及时补充通用研发图纸。

    升级 HQ 与 Repair Room 均依赖通用图纸，储备耗尽会阻断研发进度。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        state_manager.set("tasks", "blueprint_warning", True, save=True)
        logger.warning(
            "[BlueprintCheck] ⚠️ 图纸储备不足！请及时补充通用研发图纸（HQ/Repair Room 升级需要）。"
        )
        return True
