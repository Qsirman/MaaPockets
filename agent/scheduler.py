"""
scheduler.py — 口袋奇兵 AFK 系统 schedule 调度中枢

职责：
  1. 注册所有业务模块的定时任务（雷达/合并/外勤/每日/体力预测）
  2. 根据 interface.json 传入的运营强度参数动态调整调度频率
  3. 利用 StateManager.predict_stamina() 精准唤醒雷达任务
  4. 提供线程安全的 Tasker 引用管理（AgentServer 初始化后注入）

多进程解耦设计：
  - Scheduler 运行在独立线程（schedule 主循环），与 AgentServer IPC 线程解耦
  - 通过 threading.Event 实现优雅停机：stop_event.set() → 循环退出
  - 通过 tasker_ref 弱引用避免循环持有，防止内存泄漏
"""

import time
import logging
import threading
from typing import Optional

import schedule

from state_manager import state_manager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 调度频率配置（单位：分钟）
# 运营强度 → (雷达间隔, 合并间隔, 外勤间隔)
# ──────────────────────────────────────────────
INTENSITY_CONFIG: dict = {
    "轻度":  {"radar": 60,  "merge": 120, "expedition": 180},
    "标准":  {"radar": 30,  "merge": 60,  "expedition": 120},
    "激进":  {"radar": 15,  "merge": 30,  "expedition": 60},
}
DEFAULT_INTENSITY = "标准"


class TopWarScheduler:
    """
    口袋奇兵 AFK 调度中枢。

    使用流程：
      1. 在 AgentServer 启动后，调用 inject_tasker(tasker) 注入 Tasker 实例
      2. 调用 start(intensity) 启动调度循环（阻塞线程，建议在独立线程运行）
      3. 调用 stop() 优雅停机
    """

    def __init__(self):
        self._tasker = None          # MaaTasker 实例，由 AgentServer 回调后注入
        self._tasker_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._intensity = DEFAULT_INTENSITY

    # ──────────────────────────────────────────────
    # Tasker 注入接口
    # ──────────────────────────────────────────────

    def inject_tasker(self, tasker) -> None:
        """
        由 AgentServer 的回调函数调用，注入 Tasker 实例。
        Tasker 是 MaaFramework 的任务执行器，负责 run_task()。

        线程安全设计：
          AgentServer 可能在任意线程回调此方法，
          使用 Lock 保证 _tasker 赋值的原子性。
        """
        with self._tasker_lock:
            self._tasker = tasker
        logger.info("[Scheduler] Tasker 注入成功，调度器可以开始执行任务。")

    def _run_task_safe(self, task_name: str, pipeline_override: Optional[dict] = None) -> bool:
        """
        线程安全地调用 tasker.run_task()。
        若 tasker 尚未注入或任务运行失败，记录日志并返回 False。

        参数：
          task_name        — Pipeline 中的入口节点名
          pipeline_override — 可选的参数覆盖字典
        """
        with self._tasker_lock:
            tasker = self._tasker
        if tasker is None:
            logger.warning("[Scheduler] Tasker 尚未就绪，跳过任务: %s", task_name)
            return False
        try:
            logger.info("[Scheduler] 开始执行任务: %s", task_name)
            result = tasker.run_task(task_name, pipeline_override or {})
            if result:
                logger.info("[Scheduler] 任务完成: %s", task_name)
            else:
                logger.warning("[Scheduler] 任务失败: %s", task_name)
            return result
        except Exception as e:
            logger.error("[Scheduler] 任务异常: %s — %s", task_name, e, exc_info=True)
            return False

    # ──────────────────────────────────────────────
    # 各业务任务回调
    # ──────────────────────────────────────────────

    def _job_stamina_check(self) -> None:
        """
        体力预测 tick（每 3 分钟执行一次）。
        若体力预测接近满值（>=70/75），且当前雷达任务不处于冷却期，
        则立即触发雷达任务（提前唤醒），并重置雷达调度的下次执行时间。

        设计初衷：
          避免体力溢出浪费——体力满后不再恢复，
          脚本必须在体力即将满时精准触发雷达任务消耗体力。
        """
        predicted = state_manager.predict_stamina()
        logger.debug("[Scheduler] 体力预测: %d/75", predicted)

        # 体力 >= 70 视为"即将满体力"，提前触发雷达任务
        if predicted >= 70:
            logger.info("[Scheduler] 体力预测 %d>=70，触发雷达任务（体力优先唤醒）。", predicted)
            self._job_radar()

    def _job_radar(self) -> None:
        """
        雷达清场任务。
        执行前检查：体力是否足够（预测值 >= 5 才有意义）。
        执行后更新 StateManager 中的 last_radar_run 时间戳。
        """
        predicted = state_manager.predict_stamina()
        if predicted < 5:
            logger.info("[Scheduler] 体力不足（%d<5），跳过雷达任务。", predicted)
            return

        # 构建 pipeline_override，从 StateManager 读取当前体力药水策略
        allow_potion = state_manager.get("tasks", "top_lord_active", default=False)
        override = {
            "StartRadar": {
                "custom_recognition_param": {
                    "allow_potion": allow_potion,
                    "use_free_refresh": True,
                }
            }
        }
        if self._run_task_safe("StartRadar", override):
            state_manager.set("scheduler", "last_radar_run", time.time())

    def _job_merge(self) -> None:
        """
        合并引擎任务。扫描基建和兵团区域，执行同等级单位合并。
        """
        if self._run_task_safe("StartMerge"):
            state_manager.set("scheduler", "last_merge_run", time.time())

    def _job_expedition(self) -> None:
        """
        外勤派遣任务。包括大地图搜索、卡车护送、背包安全卸载。
        """
        quality = "SR"  # 默认 SR，可由 interface.json option 覆盖
        protect = True
        override = {
            "StartExpedition": {
                "custom_recognition_param": {
                    "truck_min_quality": quality,
                    "inventory_protect": protect,
                }
            }
        }
        if self._run_task_safe("StartExpedition", override):
            state_manager.set("scheduler", "last_expedition_run", time.time())

    def _job_daily(self) -> None:
        """
        每日常规任务：捐赠、科研、岛屿大作战。
        仅在今日任务未完成时执行（通过 StateManager 标记判断）。
        """
        # 检查跨日重置
        state_manager.check_and_reset_daily()

        donated = state_manager.get("tasks", "donated", default=False)
        if donated:
            logger.info("[Scheduler] 今日每日任务已完成，跳过。")
            return

        if self._run_task_safe("StartDailyTasks"):
            state_manager.set("scheduler", "last_daily_run", time.time())

    # ──────────────────────────────────────────────
    # 启动 / 停机
    # ──────────────────────────────────────────────

    def start(self, intensity: str = DEFAULT_INTENSITY) -> None:
        """
        启动调度主循环（阻塞当前线程）。
        建议在 threading.Thread 中运行此方法。

        参数：
          intensity — 运营强度（"轻度"/"标准"/"激进"），决定调度间隔
        """
        self._intensity = intensity
        cfg = INTENSITY_CONFIG.get(intensity, INTENSITY_CONFIG[DEFAULT_INTENSITY])

        logger.info(
            "[Scheduler] 启动调度循环，运营强度: %s（雷达=%dmin, 合并=%dmin, 外勤=%dmin）",
            intensity, cfg["radar"], cfg["merge"], cfg["expedition"],
        )

        # 清空旧调度计划，防止重复注册
        schedule.clear()

        # 体力预测 tick — 固定 3 分钟检查一次（与体力恢复周期对齐）
        schedule.every(3).minutes.do(self._job_stamina_check)

        # 雷达清场 — 按运营强度设置间隔
        schedule.every(cfg["radar"]).minutes.do(self._job_radar)

        # 合并引擎 — 按运营强度设置间隔
        schedule.every(cfg["merge"]).minutes.do(self._job_merge)

        # 外勤派遣 — 按运营强度设置间隔
        schedule.every(cfg["expedition"]).minutes.do(self._job_expedition)

        # 每日任务 — 每天 08:00 执行一次，同时启动时立即检查一次
        schedule.every().day.at("08:00").do(self._job_daily)

        # 启动时立即触发一次每日检查（处理重启后的状态恢复）
        self._job_daily()

        # 调度主循环
        self._stop_event.clear()
        while not self._stop_event.is_set():
            schedule.run_pending()
            # 每秒检查一次待执行任务，平衡响应性与 CPU 占用
            self._stop_event.wait(timeout=1.0)

        logger.info("[Scheduler] 调度循环已停止。")

    def stop(self) -> None:
        """
        优雅停机：设置 stop_event，让 start() 的主循环退出。
        调用后等待循环自然结束（最多 2 秒延迟）。
        """
        logger.info("[Scheduler] 收到停机信号。")
        self._stop_event.set()
        schedule.clear()


# ──────────────────────────────────────────────
# 模块级单例
# ──────────────────────────────────────────────
scheduler = TopWarScheduler()
