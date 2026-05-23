import importlib.util
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.game import BomberEnv


def load_agent_instance(agent_path: str, agent_id: int):
    # Add the submission directory to sys.path so that helper modules bundled
    # alongside agent.py (e.g. reward.py, utils.py, model.py) can be imported
    # normally with plain `import reward` or `from utils import ...`.
    # We insert at position 0 so submission-local modules take priority over any
    # global package with the same name, and we avoid adding duplicates.
    agent_dir = str(Path(agent_path).parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    # Use the module name "agent" (matching the filename) so that helper modules
    # which do `from agent import X` find THIS module in sys.modules instead of
    # triggering a fresh re-load.  Pre-registering before exec_module is the
    # standard Python pattern for breaking circular imports: subsequent imports
    # of "agent" during execution will receive the partially-initialized module
    # rather than starting a second, conflicting load of agent.py.
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module   # pre-register before exec to break circular imports
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("agent", None)  # clean up so the next load starts fresh
        raise

    if hasattr(module, "Agent") and isinstance(getattr(module, "Agent"), type):
        agent_cls = getattr(module, "Agent")
    else:
        agent_cls = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith("Agent"):
                agent_cls = attr
                break
        if agent_cls is None:
            raise AttributeError(
                "No valid agent class found. Expected 'Agent' or a class ending with 'Agent'."
            )

    try:
        return agent_cls(agent_id)
    except TypeError:
        return agent_cls()


def sanitize_action(action) -> tuple[int, bool]:
    try:
        value = int(action)
    except Exception:
        return 0, False

    if 0 <= value <= 5:
        return value, True
    return 0, False


def _agent_worker(agent_path: str, agent_id: int, recv_conn, send_conn):
    import os
    import pwd
    
    # only works if the parent process was started with sudo/root.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            # 'nobody' is the standard Linux user with zero permissions
            nobody = pwd.getpwnam('nobody')
            os.setgroups([]) # Strip auxiliary groups
            os.setgid(nobody.pw_gid) # Drop group privileges
            os.setuid(nobody.pw_uid) # drop user privileges
        except Exception as exc:
            send_conn.send({"ok": False, "error": f"sandbox_failed:{exc}"})
            return

    # Force single-threaded execution to prevent thread-thrashing & CPU saturation DoS attacks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        agent = load_agent_instance(agent_path=agent_path, agent_id=agent_id)
    except Exception as exc:
        send_conn.send({"ok": False, "error": f"load_failed:{exc}"})
        return

    send_conn.send({"ok": True, "ready": True})

    while True:
        try:
            payload = recv_conn.recv()
        except EOFError:
            return

        cmd = payload.get("cmd")
        if cmd == "close":
            return

        if cmd != "act":
            send_conn.send({"ok": False, "error": f"unknown_cmd:{cmd}"})
            continue

        obs = payload.get("obs")
        try:
            action = agent.act(obs)
            send_conn.send({"ok": True, "action": action})
        except Exception as exc:
            send_conn.send({"ok": False, "error": f"act_failed:{exc}"})


@dataclass
class AgentStepResult:
    action: int
    timeout: bool = False
    error: Optional[str] = None
    invalid_action: bool = False


class AgentProcessExecutor:
    def __init__(self, agent_path: str, agent_id: int):
        self.agent_path = str(Path(agent_path))
        self.agent_id = int(agent_id)
        self.ctx = mp.get_context("fork")

        self._proc = None
        self._parent_recv = None
        self._parent_send = None

    def start(self, startup_timeout_s: float = 2.0):
        if self._proc is not None and self._proc.is_alive():
            return

        parent_recv, child_send = self.ctx.Pipe(duplex=False)
        child_recv, parent_send = self.ctx.Pipe(duplex=False)

        proc = self.ctx.Process(
            target=_agent_worker,
            args=(self.agent_path, self.agent_id, child_recv, child_send),
            daemon=True,
        )
        proc.start()

        self._proc = proc
        self._parent_recv = parent_recv
        self._parent_send = parent_send

        if not self._parent_recv.poll(startup_timeout_s):
            self.terminate()
            raise TimeoutError("agent_worker_start_timeout")

        boot = self._parent_recv.recv()
        if not boot.get("ok"):
            self.terminate()
            raise RuntimeError(boot.get("error", "agent_worker_start_failed"))

    def act_with_timeout(self, obs, timeout_s: float) -> AgentStepResult:
        if self._proc is None or not self._proc.is_alive():
            try:
                self.start()
            except Exception as exc:
                return AgentStepResult(action=0, error=f"worker_start_failed:{exc}")

        try:
            self._parent_send.send({"cmd": "act", "obs": obs})
        except Exception as exc:
            self.terminate()
            return AgentStepResult(action=0, error=f"send_failed:{exc}")

        if not self._parent_recv.poll(timeout_s):
            self.terminate()
            return AgentStepResult(action=0, timeout=True, error="act_timeout")

        try:
            reply = self._parent_recv.recv()
        except Exception as exc:
            self.terminate()
            return AgentStepResult(action=0, error=f"recv_failed:{exc}")

        if not reply.get("ok"):
            self.terminate()
            return AgentStepResult(action=0, error=reply.get("error", "act_failed"))

        action, valid = sanitize_action(reply.get("action"))
        if not valid:
            return AgentStepResult(action=0, invalid_action=True, error="invalid_action")
        return AgentStepResult(action=action)

    def terminate(self):
        if self._parent_send is not None:
            try:
                self._parent_send.send({"cmd": "close"})
            except Exception:
                pass

        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=0.2)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=0.2)

        self._proc = None
        self._parent_recv = None
        self._parent_send = None


def build_precheck_observation(seed: int = 0):
    env = BomberEnv(seed=seed, max_steps=2)
    return env.reset(seed=seed)


def runtime_precheck(
    agent_path: str, timeout_s: float = 0.1, startup_timeout_s: Optional[float] = None
) -> tuple[bool, str]:
    obs = build_precheck_observation(seed=0)
    executor = AgentProcessExecutor(agent_path=agent_path, agent_id=0)

    started = time.perf_counter()
    try:
        # Allow caller to override startup timeout (time to import/load model).
        if startup_timeout_s is None:
            startup_timeout_s = max(1.0, timeout_s * 10)
        executor.start(startup_timeout_s=startup_timeout_s)
        result = executor.act_with_timeout(obs=obs, timeout_s=timeout_s)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.timeout:
            return False, "runtime_precheck_timeout"
        if result.error and result.error != "invalid_action":
            return False, f"runtime_precheck_error:{result.error}"
        if result.invalid_action:
            return False, "runtime_precheck_invalid_action"

        return True, f"runtime_precheck_ok:{elapsed_ms:.2f}ms"
    except Exception as exc:
        return False, f"runtime_precheck_exception:{exc}"
    finally:
        executor.terminate()
