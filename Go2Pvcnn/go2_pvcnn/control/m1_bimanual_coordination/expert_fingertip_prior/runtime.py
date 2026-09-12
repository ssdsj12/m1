"""Fail-closed runtime adapter for a validated frozen O6 fingertip prior."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from typing import Any
import weakref
import torch
from ..contracts import BimanualPhase
from .artifact import load_student_artifact
from .contracts import MIXTURE_COMPONENTS, MODEL_INPUT_DIM, PHASE_ORDER, PRIOR_HORIZON, PriorPhase

_SAFE = frozenset({BimanualPhase.DONE,BimanualPhase.HOLD_SAFE,BimanualPhase.LOWER_SAFE,BimanualPhase.SAFE_RELEASE,BimanualPhase.TERMINATED})
_PHASE = {BimanualPhase.APPROACH:PriorPhase.APPROACH,BimanualPhase.PRELOAD:PriorPhase.PRELOAD,BimanualPhase.GRASP:PriorPhase.GRASP,BimanualPhase.LIFT:PriorPhase.MANIPULATE,BimanualPhase.HOLD:PriorPhase.HOLD,BimanualPhase.LOWER:PriorPhase.MANIPULATE,BimanualPhase.RELEASE:PriorPhase.RELEASE}
def _pos(name: str, value: object) -> float:
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or float(value)<=0: raise ValueError(f"{name} must be finite and positive")
    return float(value)
def _ms(start: int) -> float: return max(0.,(time.perf_counter_ns()-start)/1e6)
@dataclass(frozen=True)
class PriorRuntimeCfg:
    log_std_min: float=-7.; log_std_max: float=3.; precision_min: float=1e-4; precision_max: float=1e4; logit_weight: float=1e-2; inference_timeout_ms: float=20.; safe_phases: frozenset[BimanualPhase]=field(default_factory=lambda:_SAFE)
    def __post_init__(self):
        if _pos("precision_min",self.precision_min)>_pos("precision_max",self.precision_max): raise ValueError("precision_min must not exceed precision_max")
        for name in ("log_std_min","log_std_max"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)): raise ValueError(f"{name} must be finite")
        if self.log_std_min>self.log_std_max: raise ValueError("log_std_min must not exceed log_std_max")
        _pos("logit_weight",self.logit_weight); _pos("inference_timeout_ms",self.inference_timeout_ms)
        if not isinstance(self.safe_phases,frozenset) or not self.safe_phases or not all(isinstance(x,BimanualPhase) for x in self.safe_phases): raise TypeError("safe_phases must be non-empty BimanualPhase frozenset")
@dataclass(frozen=True)
class O6FingertipPriorInput: fingertip_positions_b: torch.Tensor; contact_jacobian: torch.Tensor; qd: torch.Tensor; contact_mask: torch.Tensor; phase: BimanualPhase
@dataclass(frozen=True)
class FingertipPriorTarget:
    mean_velocity: torch.Tensor; precision: torch.Tensor; component: int; probability: float
    def __post_init__(self):
        for value in (self.mean_velocity,self.precision):
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or value.device.type!="cpu" or value.shape!=(15,) or not torch.isfinite(value).all().item(): raise ValueError("target tensors must be finite CPU float64 (15,)")
        if not torch.all(self.precision>=0).item() or type(self.component) is not int or not 0<=self.component<MIXTURE_COMPONENTS or not isinstance(self.probability,float) or not math.isfinite(self.probability) or not 0<=self.probability<=1: raise ValueError("invalid target")
        object.__setattr__(self,"mean_velocity",self.mean_velocity.clone()); object.__setattr__(self,"precision",self.precision.clone())
@dataclass(frozen=True)
class FingertipPriorDiagnostics:
    enabled: bool; reason: str|None; inference_ms: float
    def __post_init__(self):
        if not isinstance(self.enabled,bool) or self.enabled != (self.reason is None) or not isinstance(self.inference_ms,float) or not math.isfinite(self.inference_ms) or self.inference_ms<0: raise ValueError("invalid diagnostics")
@dataclass(frozen=True)
class PriorQueryResult:
    target: FingertipPriorTarget|None; diagnostics: FingertipPriorDiagnostics
    def __post_init__(self):
        if not isinstance(self.diagnostics,FingertipPriorDiagnostics) or self.diagnostics.enabled!=(self.target is not None): raise ValueError("target and diagnostics disagree")
def _off(reason: str, elapsed: float=0.) -> PriorQueryResult: return PriorQueryResult(None,FingertipPriorDiagnostics(False,reason,float(elapsed)))
def _tensor(v: object, shape: tuple[int,...], dtype: torch.dtype) -> bool: return isinstance(v,torch.Tensor) and v.shape==shape and v.dtype==dtype and v.device.type=="cpu" and bool(torch.isfinite(v).all().item())
def _worker(model: torch.nn.Module, requests: Any, responses: Any, ready: Any) -> None:
    try: model.eval(); ready.put(True)
    except BaseException: ready.put(False); return
    while True:
        request=requests.get()
        if request is None: return
        ident,value=request; start=time.perf_counter_ns()
        try:
            with torch.no_grad():
                out=model(value); payload=(out.logits.detach().cpu(),out.mean.detach().cpu(),out.log_std.detach().cpu())
            responses.put((ident,"ok",_ms(start),payload))
        except BaseException as error: responses.put((ident,"exception",_ms(start),type(error).__name__))
class _ReaperController:
    """Owns every spawned worker until it is demonstrably dead and its slot is released."""
    def __init__(self, *, capacity: int):
        if type(capacity) is not int or capacity < 1: raise ValueError("reaper capacity must be positive")
        self._slots=threading.BoundedSemaphore(capacity)
        self._queue: queue.Queue[Any]=queue.Queue(maxsize=capacity)
        self._lock=threading.Lock()
        self._thread: threading.Thread|None=None
        self._backoff=threading.Event()
    def acquire(self) -> bool: return self._slots.acquire(blocking=False)
    def release(self) -> None: self._slots.release()
    def enqueue(self, worker: Any) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread=threading.Thread(target=self._loop,name="fingertip-prior-reaper",daemon=True)
                self._thread.start()
        self._queue.put_nowait(worker)
    def _loop(self) -> None:
        while True:
            worker=self._queue.get()
            try:
                if not worker._reap_terminated():
                    self._backoff.wait(timeout=0.01)
                    self.enqueue(worker)
            except BaseException:
                self._backoff.wait(timeout=0.01)
                self.enqueue(worker)
            finally:
                self._queue.task_done()


_REAPER=_ReaperController(capacity=8)


class _Worker:
    _STARTUP_TIMEOUT_S = 5.0
    _GRACEFUL_JOIN_S = 0.1
    _FORCED_JOIN_S = 0.1

    def __init__(self, model: torch.nn.Module):
        self._owner=_REAPER
        self._slot_owned=False
        self._closed=False
        self.reaped=threading.Event()
        self.requests: Any=None
        self.responses: Any=None
        self.process: Any=None
        if not self._owner.acquire(): raise RuntimeError("prior worker capacity exhausted")
        self._slot_owned=True
        try:
            context=mp.get_context("spawn")
            copied=copy.deepcopy(model).to("cpu")
            copied.eval()
            self.requests=context.Queue(1); self.responses=context.Queue(1); ready=context.Queue(1)
            self.process=context.Process(target=_worker,args=(copied,self.requests,self.responses,ready),daemon=True)
            self.process.start()
            if ready.get(timeout=self._STARTUP_TIMEOUT_S) is not True: raise RuntimeError("worker did not become ready")
        except BaseException as error:
            try: self.close()
            except BaseException: pass
            raise RuntimeError("prior worker initialization failed") from error

    def _dead(self) -> bool:
        if self.process is None: return True
        try: return not bool(self.process.is_alive())
        except BaseException: return False
    def _complete_reap(self) -> bool:
        if not self._dead(): return False
        for channel in (self.requests,self.responses):
            if channel is not None:
                try: channel.cancel_join_thread(); channel.close()
                except BaseException: pass
        if self._slot_owned:
            self._owner.release(); self._slot_owned=False
        self.reaped.set()
        return True
    def _request_reap(self) -> None:
        self._owner.enqueue(self)
    def close(self):
        if self.reaped.is_set(): return
        self._closed=True
        if self._dead(): self._complete_reap(); return
        try: self.requests.put_nowait(None)
        except BaseException: pass
        try: self.process.join(timeout=self._GRACEFUL_JOIN_S)
        except BaseException: pass
        if self._dead(): self._complete_reap(); return
        try: self.process.terminate()
        except BaseException: pass
        try: self.process.join(timeout=self._FORCED_JOIN_S)
        except BaseException: pass
        if self._dead(): self._complete_reap(); return
        if hasattr(self.process,"kill"):
            try: self.process.kill()
            except BaseException: pass
            try: self.process.join(timeout=self._FORCED_JOIN_S)
            except BaseException: pass
        if self._dead(): self._complete_reap(); return
        self._request_reap()
        raise RuntimeError("prior worker pending asynchronous reap")
    def detach_for_target(self) -> bool:
        self._closed=True
        cleanup_error=False
        if not self._dead():
            try: self.process.terminate()
            except BaseException: cleanup_error=True
        self._request_reap()
        return cleanup_error
    def _reap_terminated(self) -> bool:
        if self._complete_reap(): return True
        try: self.process.join(timeout=self._FORCED_JOIN_S)
        except BaseException: pass
        if self._complete_reap(): return True
        try: self.process.terminate()
        except BaseException: pass
        try: self.process.join(timeout=self._FORCED_JOIN_S)
        except BaseException: pass
        if self._complete_reap(): return True
        if hasattr(self.process,"kill"):
            try: self.process.kill()
            except BaseException: pass
            try: self.process.join(timeout=self._FORCED_JOIN_S)
            except BaseException: pass
        return self._complete_reap()


def _finalize_worker(worker: _Worker) -> None:
    try:
        worker.close()
    except BaseException:
        pass


class FrozenO6FingertipPrior:
    """Production construction is restricted to from_artifact."""
    def __init__(self,*args,**kwargs): raise TypeError("use FrozenO6FingertipPrior.from_artifact")
    @classmethod
    def from_artifact(cls,path: str|Path,*,cfg: PriorRuntimeCfg|None=None):
        loaded=load_student_artifact(path)
        if loaded.metrics.get("production_approved") is not True: raise ValueError("runtime requires a production_approved student artifact")
        if cfg is not None and not isinstance(cfg,PriorRuntimeCfg): raise TypeError("cfg must be PriorRuntimeCfg")
        worker=_Worker(loaded.model)
        self=object.__new__(cls)
        self.cfg=PriorRuntimeCfg() if cfg is None else cfg
        self._worker=worker
        self._lock=threading.Lock()
        self._id=0
        self._poisoned=None
        self._closed=False
        self._finalizer=weakref.finalize(self,_finalize_worker,worker)
        return self
    def close(self):
        with self._lock:
            if self._closed:
                if self._worker is not None:self._finalizer()
                return
            self._closed=True
            self._poisoned=("closed",0.)
            self._finalizer()
    def __enter__(self): return self
    def __exit__(self,*unused): self.close(); return False
    def _poison_async(self,reason: str,elapsed: float):
        self._closed=True
        self._poisoned=(reason,float(elapsed))
        worker=self._worker
        if worker is None:return _off(reason,elapsed)
        try:
            cleanup_error=worker.detach_for_target()
            self._worker=None
            self._finalizer.detach()
        except BaseException:
            cleanup_error=True
        final_reason=f"{reason}_cleanup_error" if cleanup_error else reason
        self._poisoned=(final_reason,float(elapsed))
        return _off(final_reason,elapsed)
    def _input(self,s: O6FingertipPriorInput):
        if not isinstance(s,O6FingertipPriorInput) or not _tensor(s.fingertip_positions_b,(5,3),torch.float64) or not _tensor(s.contact_jacobian,(15,6),torch.float64) or not _tensor(s.qd,(6,),torch.float64) or not isinstance(s.contact_mask,torch.Tensor) or s.contact_mask.shape!=(5,) or s.contact_mask.dtype!=torch.bool or s.contact_mask.device.type!="cpu" or not isinstance(s.phase,BimanualPhase): return None
        phase=_PHASE.get(s.phase)
        if phase is None:return None
        hot=torch.nn.functional.one_hot(torch.tensor(int(phase)),num_classes=len(PHASE_ORDER)).to(torch.float64)
        v=torch.cat((s.fingertip_positions_b.reshape(-1),s.contact_jacobian@s.qd,s.contact_mask.to(torch.float64),hot))
        return v.float().unsqueeze(0) if v.shape==(MODEL_INPUT_DIM,) and torch.isfinite(v).all().item() else None
    def target(self,sample: O6FingertipPriorInput,baseline_qd: torch.Tensor):
        start=time.perf_counter_ns()
        if not self._lock.acquire(timeout=self.cfg.inference_timeout_ms/1000):
            return _off("timeout",_ms(start))
        try:
            if self._poisoned is not None:return _off(*self._poisoned)
            if self._closed:return _off("closed")
            if isinstance(sample,O6FingertipPriorInput) and isinstance(sample.phase,BimanualPhase) and sample.phase in self.cfg.safe_phases:return _off("safe_phase")
            value=self._input(sample)
            if value is None or not _tensor(baseline_qd,(6,),torch.float64):return _off("invalid_input")
            elapsed=_ms(start)
            if elapsed>self.cfg.inference_timeout_ms:return self._poison_async("timeout",elapsed)
            ident=self._id; self._id+=1
            try: self._worker.requests.put_nowait((ident,value)); response=self._worker.responses.get(timeout=max(0.,(self.cfg.inference_timeout_ms-elapsed)/1000))
            except queue.Empty:return self._poison_async("timeout",_ms(start))
            except BaseException:return self._poison_async("prior_exception",_ms(start))
            elapsed=_ms(start)
            if not isinstance(response,tuple) or len(response)!=4 or response[0]!=ident:return self._poison_async("prior_exception",elapsed)
            if isinstance(response[2],(int,float)) and not isinstance(response[2],bool) and math.isfinite(float(response[2])) and response[2]>=0:elapsed=max(elapsed,float(response[2]))
            if elapsed>self.cfg.inference_timeout_ms:return self._poison_async("timeout",elapsed)
            if response[1]!="ok":return self._poison_async("prior_exception",elapsed)
            payload=response[3]
            if not isinstance(payload,tuple) or len(payload)!=3:return self._poison_async("invalid_prior",elapsed)
            logits,means,std=payload; expect=(1,MIXTURE_COMPONENTS,PRIOR_HORIZON,5,3)
            if any(not _tensor(x,shape,torch.float32) for x,shape in ((logits,(1,MIXTURE_COMPONENTS)),(means,expect),(std,expect))):return self._poison_async("nonfinite_prior",elapsed)
            precision=torch.exp(-2*std[:,:,0].clamp(self.cfg.log_std_min,self.cfg.log_std_max)).clamp(self.cfg.precision_min,self.cfg.precision_max)
            if not torch.isfinite(precision).all().item() or not torch.all(precision>=self.cfg.precision_min).item() or not torch.all(precision<=self.cfg.precision_max).item():return self._poison_async("invalid_precision",elapsed)
            precision[:,:,sample.contact_mask,:]=0
            mean=means[:,:,0].reshape(MIXTURE_COMPONENTS,15).double(); precision=precision.reshape(MIXTURE_COMPONENTS,15).double(); probs=logits[0].log_softmax(-1).exp()
            if not torch.isfinite(probs).all().item() or not torch.all(probs>=0).item() or not torch.isclose(probs.sum(),torch.tensor(1.,dtype=torch.float32),atol=1e-6,rtol=0).item():return self._poison_async("invalid_probabilities",elapsed)
            score=(((sample.contact_jacobian@baseline_qd).reshape(1,15)-mean).square()*precision).sum(1)-float(self.cfg.logit_weight)*logits[0].log_softmax(-1).double()
            elapsed=_ms(start)
            if elapsed>self.cfg.inference_timeout_ms:return self._poison_async("timeout",elapsed)
            if not torch.isfinite(score).all().item():return self._poison_async("invalid_precision",elapsed)
            component=int(torch.argmin(score).item())
            return PriorQueryResult(FingertipPriorTarget(mean[component],precision[component],component,float(probs[component].item())),FingertipPriorDiagnostics(True,None,elapsed))
        finally:
            self._lock.release()
__all__=["FingertipPriorDiagnostics","FingertipPriorTarget","FrozenO6FingertipPrior","O6FingertipPriorInput","PriorQueryResult","PriorRuntimeCfg"]
