"""
state_manager.py — 口袋奇兵 AFK 系统状态机核心

职责：
  1. 维护全局运营状态字典（体力、雷达、背包、每日任务标记等）
  2. 通过原子落盘技术（写 .tmp → os.replace）防范断电导致的 JSON 截断崩溃
  3. 提供离线体力预测方法，避免频繁启动游戏查看体力
  4. 提供线程安全的读写接口（使用 threading.Lock）

状态文件路径：topwar_state.json（与 agent/main.py 同目录）
"""

import json
import os
import time
import threading
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 常量定义
# ──────────────────────────────────────────────
STAMINA_MAX: int = 75           # 游戏体力上限
STAMINA_REGEN_SEC: float = 180.0  # 每点体力恢复所需秒数（3 分钟）
STATE_FILE: Path = Path(__file__).parent / "topwar_state.json"


# ──────────────────────────────────────────────
# 默认状态结构
# ──────────────────────────────────────────────
def _default_state() -> dict:
    """
    返回全新的默认状态字典。
    当状态文件损坏或不存在时，以此作为初始值。
    """
    now = time.time()
    return {
        # ── 体力系统 ──────────────────────────────────────────
        "stamina": {
            "value": 0,           # 上次记录的体力值（游戏内实测或初始为0）
            "last_updated": now,  # 上次更新的 Unix 时间戳（用于离线推演）
        },
        # ── 雷达系统 ──────────────────────────────────────────
        "radar": {
            "refreshes_left": 5,          # 今日剩余免费刷新次数
            "last_reset_date": "",        # 上次重置日期 "YYYY-MM-DD"，用于跨日判断
        },
        # ── 背包系统 ──────────────────────────────────────────
        "inventory": {
            "potions": 0,          # 体力药水数量（通过 OCR 识别更新）
            "resource_shortage": False,  # 是否触发「资源不足」警报（用于解锁盲盒）
        },
        # ── 每日任务标记 ──────────────────────────────────────
        "tasks": {
            "donated": False,           # 今日是否已完成捐赠
            "researched": False,        # 今日是否已触发科研
            "island_battle_done": False,  # 今日岛屿大作战是否已完成扫荡
            "top_lord_active": False,   # Top Lord（最强领主）活动是否处于激活期
            "blueprint_warning": False, # 图纸不足预警
        },
        # ── 断线重连追踪 ──────────────────────────────────────
        "reconnect": {
            "fail_count": 0,       # 连续失败次数（用于指数退避）
            "last_attempt": 0.0,   # 上次重连尝试时间戳
        },
        # ── 调度层元数据 ──────────────────────────────────────
        "scheduler": {
            "intensity": "标准",   # 运营强度：轻度 / 标准 / 激进
            "last_radar_run": 0.0,
            "last_merge_run": 0.0,
            "last_expedition_run": 0.0,
            "last_daily_run": 0.0,
        },
    }


class StateManager:
    """
    口袋奇兵 AFK 系统全局状态机。

    所有业务模块通过此类的单例实例读写状态，无需直接操作 JSON 文件。
    内部使用 threading.Lock 保证多线程环境下的读写原子性。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {}
        self.load_state()

    # ──────────────────────────────────────────────
    # 持久化：加载与保存
    # ──────────────────────────────────────────────

    def load_state(self) -> None:
        """
        从磁盘加载状态文件。
        若文件不存在或 JSON 损坏，回滚至默认状态并立即落盘，
        确保系统始终有合法的状态基线。
        """
        with self._lock:
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    # 深度合并：以默认值为基，loaded 覆盖存在的键，
                    # 防止版本升级后新增字段缺失导致 KeyError
                    self._state = self._deep_merge(_default_state(), loaded)
                    logger.info("[StateManager] 状态文件加载成功: %s", STATE_FILE)
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(
                        "[StateManager] 状态文件损坏，回滚默认值。错误: %s", e
                    )
                    self._state = _default_state()
                    self._write_file()  # 立即用默认值修复磁盘文件
            else:
                logger.info("[StateManager] 状态文件不存在，初始化默认状态。")
                self._state = _default_state()
                self._write_file()

    def save_state_atomic(self) -> None:
        """
        原子落盘：先写入 .tmp 临时文件，再通过 os.replace() 原子重命名覆盖目标。

        设计初衷：
          - 直接写目标文件若中途断电，会产生截断的 JSON 导致下次启动崩溃。
          - os.replace() 在同一文件系统内是原子操作（POSIX rename 语义），
            保证磁盘上始终存在完整的 JSON，不存在"写了一半"的中间态。
        """
        with self._lock:
            self._write_file()

    def _write_file(self) -> None:
        """
        内部写文件逻辑（调用方须已持有锁）。
        使用 tmp 文件 + os.replace 原子写入。
        """
        tmp_path = STATE_FILE.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 强制刷到硬件，防止 OS 缓存层截断
            os.replace(tmp_path, STATE_FILE)  # 原子重命名
        except OSError as e:
            logger.error("[StateManager] 原子写入失败: %s", e)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ──────────────────────────────────────────────
    # 读写接口（线程安全）
    # ──────────────────────────────────────────────

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        线程安全地读取嵌套键值。
        示例：manager.get("stamina", "value")
        """
        with self._lock:
            node = self._state
            for k in keys:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    return default
            return node

    def set(self, *keys_and_value, save: bool = True) -> None:
        """
        线程安全地写入嵌套键值，并可选择是否立即落盘。
        示例：manager.set("stamina", "value", 50)
               manager.set("tasks", "donated", True, save=True)

        参数 save=True 会触发原子落盘，save=False 仅修改内存（批量操作时使用）。
        注意：keys_and_value 最后一个元素为 value，其余为嵌套键路径。
        """
        if len(keys_and_value) < 2:
            raise ValueError("至少需要一个键和一个值")
        *keys, value = keys_and_value
        with self._lock:
            node = self._state
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value
            if save:
                self._write_file()

    def update_section(self, section: str, data: dict, save: bool = True) -> None:
        """
        批量更新顶层 section 的多个字段，比多次 set() 调用更高效。
        示例：manager.update_section("tasks", {"donated": True, "researched": True})
        """
        with self._lock:
            if section not in self._state:
                self._state[section] = {}
            self._state[section].update(data)
            if save:
                self._write_file()

    def snapshot(self) -> dict:
        """
        返回当前状态的深拷贝（用于日志或调试，不暴露内部引用）。
        """
        import copy
        with self._lock:
            return copy.deepcopy(self._state)

    # ──────────────────────────────────────────────
    # 体力预测引擎
    # ──────────────────────────────────────────────

    def predict_stamina(self) -> int:
        """
        离线数学推演当前体力值，避免频繁进入游戏查看。

        公式：
          elapsed_sec = now - last_updated
          recovered   = int(elapsed_sec / STAMINA_REGEN_SEC)
          current     = min(last_value + recovered, STAMINA_MAX)

        返回：预测的当前体力值（整数，0 ~ STAMINA_MAX）
        """
        with self._lock:
            last_value = self._state["stamina"]["value"]
            last_updated = self._state["stamina"]["last_updated"]

        now = time.time()
        elapsed = now - last_updated
        recovered = int(elapsed / STAMINA_REGEN_SEC)
        predicted = min(last_value + recovered, STAMINA_MAX)
        logger.debug(
            "[Stamina] 预测: %d（上次=%d, 恢复=%d点, 已过%.1f分钟）",
            predicted, last_value, recovered, elapsed / 60,
        )
        return predicted

    def seconds_to_full_stamina(self) -> float:
        """
        计算距离体力回满还需多少秒。
        返回 0.0 表示当前已满体力，无需等待。
        用于 scheduler 精准调度：预知满体力时刻再触发雷达任务。
        """
        current = self.predict_stamina()
        if current >= STAMINA_MAX:
            return 0.0
        missing = STAMINA_MAX - current
        return missing * STAMINA_REGEN_SEC

    def sync_stamina_from_game(self, real_value: int) -> None:
        """
        当通过 OCR 或游戏界面获取到真实体力值时，调用此方法同步校正。
        重置 last_updated 时间戳，确保后续推演基于实测值。
        """
        self.update_section(
            "stamina",
            {"value": real_value, "last_updated": time.time()},
            save=True,
        )
        logger.info("[Stamina] 从游戏同步实测值: %d", real_value)

    # ──────────────────────────────────────────────
    # 每日重置检测
    # ──────────────────────────────────────────────

    def check_and_reset_daily(self) -> bool:
        """
        检测是否跨越了新的游戏日（以自然日为准）。
        若是，重置所有每日任务标记和雷达刷新次数，并落盘。
        返回 True 表示发生了重置。

        多进程解耦考量：
          此方法在 scheduler 调度循环的每次 tick 开始时调用，
          确保每日状态只被重置一次，且重置后立即落盘防止重复触发。
        """
        from datetime import date
        today_str = date.today().isoformat()
        last_reset = self.get("radar", "last_reset_date", default="")

        if last_reset != today_str:
            logger.info("[Daily] 检测到新的游戏日 %s，执行每日重置。", today_str)
            self.update_section("tasks", {
                "donated": False,
                "researched": False,
                "island_battle_done": False,
                "top_lord_active": False,
                "blueprint_warning": False,
            }, save=False)
            self.update_section("radar", {
                "refreshes_left": 5,
                "last_reset_date": today_str,
            }, save=True)  # 一次性落盘
            return True
        return False

    # ──────────────────────────────────────────────
    # 断线重连追踪
    # ──────────────────────────────────────────────

    def get_reconnect_backoff(self) -> float:
        """
        根据连续失败次数计算指数退避等待时间（秒）。
        公式：min(2^fail_count, 256) 秒，最长等待 256 秒（约4分钟）。

        指数退避设计初衷：
          避免网络恢复前的密集重试造成账号异常或浪费系统资源。
          首次失败等 1 秒，之后依次 2/4/8/16/32/64/128/256 秒。
        """
        fail_count = self.get("reconnect", "fail_count", default=0)
        return min(2 ** fail_count, 256)

    def record_reconnect_attempt(self, success: bool) -> None:
        """
        记录一次重连尝试结果。
        成功：清零失败计数；失败：fail_count +1 并记录时间戳。
        """
        now = time.time()
        if success:
            self.update_section("reconnect", {
                "fail_count": 0,
                "last_attempt": now,
            })
            logger.info("[Reconnect] 重连成功，已清零失败计数。")
        else:
            fail_count = self.get("reconnect", "fail_count", default=0) + 1
            self.update_section("reconnect", {
                "fail_count": fail_count,
                "last_attempt": now,
            })
            logger.warning("[Reconnect] 重连失败，累计失败次数: %d，下次退避: %.0fs",
                           fail_count, self.get_reconnect_backoff())

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        深度合并两个字典：override 的值覆盖 base 的值，
        对于字典类型的值递归合并，确保新增字段有默认值。
        """
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = StateManager._deep_merge(result[k], v)
            else:
                result[k] = v
        return result


# ──────────────────────────────────────────────
# 模块级单例（供其他模块直接 import 使用）
# ──────────────────────────────────────────────
state_manager = StateManager()
