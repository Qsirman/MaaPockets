"""
main.py — 口袋奇兵 AFK 系统 AgentServer 入口

启动流程：
  1. Toolkit 初始化（加载资源路径配置）
  2. 注册所有 CustomRecognition / CustomAction（通过 import 触发装饰器）
  3. AgentServer.start_up(socket_id) 建立 IPC 通道
  4. 在独立线程启动 schedule 调度循环
  5. AgentServer.join() 阻塞主线程，保持 IPC 服务运行
  6. 收到停机信号后 scheduler.stop() + AgentServer.shut_down()

多进程解耦设计：
  - AgentServer IPC 线程 与 schedule 调度线程 完全解耦
  - 通过 TopWarScheduler.inject_tasker() 在 AgentServer 就绪后注入 Tasker
  - 日志统一写入 agent.log，方便问题排查
"""

import sys
import logging
import threading

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

import radar_scanner
import merge_engine
import expedition
import reconnect_handler

from state_manager import state_manager
from scheduler import scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _run_scheduler(intensity: str) -> None:
    """
    在独立线程中运行 schedule 调度循环。
    intensity 参数由 interface.json 的运营强度选项传入（目前默认从 StateManager 读取）。
    """
    try:
        scheduler.start(intensity)
    except Exception as e:
        logger.error("[Main] 调度线程异常退出: %s", e, exc_info=True)


def main():
    Toolkit.init_option("./")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    # 从状态机读取运营强度（由 interface.json 的 pipeline_override 在任务开始时写入）
    intensity = state_manager.get("scheduler", "intensity", default="标准")
    logger.info("[Main] 口袋奇兵 AFK 系统启动，运营强度: %s", intensity)

    # 启动 AgentServer IPC 服务（非阻塞）
    AgentServer.start_up(socket_id)
    logger.info("[Main] AgentServer IPC 通道已建立，socket_id=%s", socket_id)

    # 在独立线程启动 schedule 调度循环（与 AgentServer 线程解耦）
    sched_thread = threading.Thread(
        target=_run_scheduler,
        args=(intensity,),
        name="TopWarScheduler",
        daemon=True,  # 主线程退出时调度线程自动停止
    )
    sched_thread.start()
    logger.info("[Main] 调度线程已启动。")

    # 主线程阻塞等待 AgentServer 收到停机信号
    AgentServer.join()
    logger.info("[Main] AgentServer 收到停机信号，开始清理。")

    # 优雅停机
    scheduler.stop()
    sched_thread.join(timeout=5.0)
    AgentServer.shut_down()
    logger.info("[Main] 系统已完全停机。")


if __name__ == "__main__":
    main()
