"""Small, deterministic semantic transport; no body state or neural model access.

Call ``InputCueQueue.advance`` at a neural-step boundary. Its transitions carry
only a categorical concept and start/stop flag; event metadata stays here. The
consumer should hold ``active_concept_id`` until the stop transition. An event's
TTL limits when its pulse may START, never the duration of an accepted pulse.

``OutputStateMachine.update`` accepts decoder scores only. ``poll_state`` reads
the cached result and marks stale data without inventing a change of need.
All clocks are simulation milliseconds. Repeated calls at the same time do not
advance anything. State dictionaries are JSON-serializable and deep copies.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Concept:
    concept_id: int
    name: str
    label: str
    direction: str


REGISTRY = MappingProxyType({c.concept_id: c for c in (
    Concept(1, "NEED_FOOD", "Нужна еда", "from_brain"),
    Concept(2, "NEED_REST", "Нужен покой", "from_brain"),
    Concept(3, "NEED_GROOMING", "Нужна чистка", "from_brain"),
    Concept(101, "FOOD_LEFT", "Еда слева", "to_brain"),
    Concept(102, "FOOD_RIGHT", "Еда справа", "to_brain"),
    Concept(103, "FOOD_AHEAD", "Еда впереди", "to_brain"),
)})


class ProtocolError(ValueError):
    """A stable rejection code is available as ``code``."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"invalid_{name}")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ProtocolError(f"invalid_{name}") from None
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ProtocolError(f"invalid_{name}")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or value.strip() != value:
        raise ProtocolError(f"invalid_{name}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ProtocolError(f"invalid_{name}")
    return value


@dataclass(frozen=True)
class Event:
    schema_version: int
    episode_id: str
    event_id: str
    direction: str
    concept_id: int
    sim_time_ms: float
    ttl_ms: float
    active: bool
    source: str
    score: float | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ProtocolError("unsupported_schema_version")
        _identifier(self.episode_id, "episode_id")
        _identifier(self.event_id, "event_id")
        if type(self.concept_id) is not int or self.concept_id not in REGISTRY:
            raise ProtocolError("unknown_concept_id")
        if self.direction != REGISTRY[self.concept_id].direction:
            raise ProtocolError("wrong_direction")
        when = _number(self.sim_time_ms, "sim_time_ms")
        ttl = _number(self.ttl_ms, "ttl_ms", positive=True)
        if not math.isfinite(when + ttl):
            raise ProtocolError("invalid_expiry")
        if type(self.active) is not bool:
            raise ProtocolError("invalid_active")
        allowed = ("human", "oracle", "schedule") if self.direction == "to_brain" else ("neural_readout",)
        if self.source not in allowed:
            raise ProtocolError("invalid_source")
        if self.direction == "to_brain":
            if not self.active:
                raise ProtocolError("unsupported_input_transition")
            if self.score is not None:
                raise ProtocolError("input_score_forbidden")
        elif self.score is not None and _number(self.score, "score") > 1:
            raise ProtocolError("invalid_score")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Event:
        if not isinstance(value, Mapping):
            raise ProtocolError("invalid_event")
        required = set(cls.__dataclass_fields__) - {"score"}
        if not required <= value.keys() or set(value) - set(cls.__dataclass_fields__):
            raise ProtocolError("invalid_event_fields")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.score is None:
            del result["score"]
        return result


@dataclass(frozen=True)
class CueTransition:
    concept_id: int
    active: bool


class InputCueQueue:
    """Bounded earliest-timestamp-first queue, arrival order breaks ties.

    Inbound ``active:false`` is deliberately unsupported: each accepted cue is
    one fixed-duration pulse, with its stop generated by this transport. TTL
    expires at ``sim_time_ms + ttl_ms`` (exclusive delivery endpoint).

    Accepted IDs remain deduplicated for the whole episode. At ``max_events``
    the episode rejects further events instead of evicting IDs and permitting
    replay. Rejections do not consume this budget. The audit journal is a ring.
    """

    def __init__(self, episode_id: str, *, pulse_ms: float = 250,
                 capacity: int = 16, max_events: int = 4096, journal_size: int = 256):
        self.pulse_ms = _number(pulse_ms, "pulse_ms", positive=True)
        self.capacity = _positive_int(capacity, "capacity")
        self.max_events = _positive_int(max_events, "max_events")
        self.journal_size = _positive_int(journal_size, "journal_size")
        self.reset(episode_id)

    def reset(self, episode_id: str, sim_time_ms: float = 0) -> None:
        episode = _identifier(episode_id, "episode_id")
        now = _number(sim_time_ms, "sim_time_ms")
        self.episode_id = episode
        self._now = now
        self._queue: list[Event] = []
        self._seen: set[str] = set()
        self._active: tuple[Event, float, float] | None = None
        self._journal: deque[dict[str, Any]] = deque(maxlen=self.journal_size)

    @property
    def active_concept_id(self) -> int | None:
        return None if self._active is None else self._active[0].concept_id

    @property
    def journal(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(list(self._journal)))

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def _clock(self, sim_time_ms: float) -> float:
        now = _number(sim_time_ms, "sim_time_ms")
        if now < self._now:
            raise ProtocolError("time_reversed")
        self._now = now
        return now

    def _log(self, status: str, event_id: str | None, reason: str | None = None) -> None:
        self._journal.append({"sim_time_ms": self._now, "event_id": event_id,
                              "status": status, "reason": reason})

    def submit(self, value: Event | Mapping[str, Any], sim_time_ms: float) -> bool:
        """Validate and queue; invalid input returns False and a bounded log entry."""
        now = self._clock(sim_time_ms)
        event_id = None
        try:
            if isinstance(value, Event):
                event = value
            else:
                # Never retain arbitrary malformed payloads in the bounded log.
                candidate_id = value.get("event_id") if isinstance(value, Mapping) else None
                try:
                    event_id = _identifier(candidate_id, "event_id")
                except ProtocolError:
                    pass
                event = Event.from_dict(value)
            event_id = event.event_id
            if event.direction != "to_brain":
                raise ProtocolError("wrong_direction")
            if event.episode_id != self.episode_id:
                raise ProtocolError("wrong_episode")
            if event.event_id in self._seen:
                raise ProtocolError("duplicate_event")
            if now >= event.sim_time_ms + event.ttl_ms:
                raise ProtocolError("expired_event")
            if len(self._seen) >= self.max_events:
                raise ProtocolError("episode_event_limit")
            if len(self._queue) >= self.capacity:
                raise ProtocolError("queue_full")
        except ProtocolError as error:
            self._log("rejected", event_id, error.code)
            return False
        self._seen.add(event.event_id)
        self._queue.append(event)
        self._queue.sort(key=lambda item: item.sim_time_ms)
        self._log("queued", event.event_id)
        return True

    def advance(self, sim_time_ms: float) -> tuple[CueTransition, ...]:
        """Deliver at most one new pulse; finish old pulses before new starts."""
        now = self._clock(sim_time_ms)
        transitions: list[CueTransition] = []
        if self._active is not None and now >= self._active[2]:
            event = self._active[0]
            transitions.append(CueTransition(event.concept_id, False))
            self._log("pulse_finished", event.event_id)
            self._active = None
        live = []
        for event in self._queue:
            if now >= event.sim_time_ms + event.ttl_ms:
                self._log("rejected", event.event_id, "expired_event")
            else:
                live.append(event)
        self._queue = live
        if self._active is None and self._queue and self._queue[0].sim_time_ms <= now:
            event = self._queue.pop(0)
            end = now + self.pulse_ms
            if not math.isfinite(end):
                raise ProtocolError("invalid_pulse_end")
            self._active = (event, now, end)
            transitions.append(CueTransition(event.concept_id, True))
            self._log("delivered", event.event_id)
        return tuple(transitions)

    def _config(self) -> dict[str, Any]:
        return {"pulse_ms": self.pulse_ms, "capacity": self.capacity,
                "max_events": self.max_events, "journal_size": self.journal_size}

    def get_state(self) -> dict[str, Any]:
        active = None if self._active is None else {
            "event": self._active[0].to_dict(), "start_ms": self._active[1], "end_ms": self._active[2]}
        return {"format": "fly_semantic.input.v1", "config": self._config(),
                "episode_id": self.episode_id, "now_ms": self._now,
                "queue": [event.to_dict() for event in self._queue],
                "seen": sorted(self._seen), "active": active, "journal": list(self.journal)}

    def set_state(self, value: Mapping[str, Any]) -> None:
        """Restore compatible state atomically; constructor configuration must match."""
        try:
            if set(value) != set(self.get_state()) or value["format"] != "fly_semantic.input.v1" or value["config"] != self._config():
                raise ValueError
            candidate = InputCueQueue(value["episode_id"], **self._config())
            candidate._now = _number(value["now_ms"], "now_ms")
            seen = value["seen"]
            if not isinstance(seen, list) or len(seen) > self.max_events:
                raise ValueError
            candidate._seen = {_identifier(item, "event_id") for item in seen}
            if len(candidate._seen) != len(seen):
                raise ValueError
            queue = value["queue"]
            if not isinstance(queue, list) or len(queue) > self.capacity:
                raise ValueError
            candidate._queue = [Event.from_dict(item) for item in queue]
            involved = list(candidate._queue)
            if candidate._queue != sorted(candidate._queue, key=lambda item: item.sim_time_ms):
                raise ValueError
            active = value["active"]
            if active is not None:
                if set(active) != {"event", "start_ms", "end_ms"}:
                    raise ValueError
                event = Event.from_dict(active["event"])
                start = _number(active["start_ms"], "start_ms")
                end = _number(active["end_ms"], "end_ms")
                if start > candidate._now or end != start + self.pulse_ms or not event.sim_time_ms <= start < event.sim_time_ms + event.ttl_ms:
                    raise ValueError
                candidate._active = (event, start, end)
                involved.append(event)
            if len({event.event_id for event in involved}) != len(involved):
                raise ValueError
            if any(event.episode_id != candidate.episode_id or event.direction != "to_brain" or event.event_id not in candidate._seen for event in involved):
                raise ValueError
            journal = value["journal"]
            if not isinstance(journal, list) or len(journal) > self.journal_size:
                raise ValueError
            for item in journal:
                if set(item) != {"sim_time_ms", "event_id", "status", "reason"}:
                    raise ValueError
                if _number(item["sim_time_ms"], "sim_time_ms") > candidate._now:
                    raise ValueError
                if item["event_id"] is not None:
                    _identifier(item["event_id"], "event_id")
                if item["status"] not in ("queued", "rejected", "delivered", "pulse_finished"):
                    raise ValueError
                if item["reason"] is not None:
                    _identifier(item["reason"], "reason")
            candidate._journal.extend(deepcopy(journal))
        except (TypeError, ValueError, KeyError, AttributeError):
            raise ProtocolError("invalid_input_checkpoint") from None
        self.__dict__.update(candidate.__dict__)


class OutputStateMachine:
    """Independent hysteresis for supported concepts, consuming only neural scores.

    Calls faster than ``sample_ms`` produce no new observation. A skipped sample
    (gap greater than sample_ms) clears pending confirmation; it is not filled
    with fabricated repeated observations. Active signals get heartbeats. An
    empty supported_concepts sequence represents feature collection before a
    readout is trained: update accepts only {}, and no output is fabricated.
    """

    def __init__(self, episode_id: str, supported_concepts: tuple[int, ...] = (1,), *,
                 on_threshold: float = 0.7, off_threshold: float = 0.4,
                 confirm_samples: int = 2, sample_ms: float = 100,
                 heartbeat_ms: float = 1000, event_ttl_ms: float = 1500):
        if not isinstance(supported_concepts, (tuple, list)) or any(type(item) is not int or item not in (1, 2, 3) for item in supported_concepts) or len(set(supported_concepts)) != len(supported_concepts):
            raise ProtocolError("invalid_supported_concepts")
        self.supported_concepts = tuple(sorted(supported_concepts))
        self.on_threshold = _number(on_threshold, "on_threshold")
        self.off_threshold = _number(off_threshold, "off_threshold")
        if not self.off_threshold < self.on_threshold <= 1:
            raise ProtocolError("invalid_thresholds")
        self.confirm_samples = _positive_int(confirm_samples, "confirm_samples")
        self.sample_ms = _number(sample_ms, "sample_ms", positive=True)
        self.heartbeat_ms = _number(heartbeat_ms, "heartbeat_ms", positive=True)
        self.event_ttl_ms = _number(event_ttl_ms, "event_ttl_ms", positive=True)
        self.reset(episode_id)

    def reset(self, episode_id: str, sim_time_ms: float = 0) -> None:
        episode = _identifier(episode_id, "episode_id")
        now = _number(sim_time_ms, "sim_time_ms")
        self.episode_id = episode
        self._now = now
        self._last_sample: float | None = None
        self._sequence = 0
        self._states = {concept: {"active": False, "pending_count": 0,
                                 "score": None, "last_event": None}
                        for concept in self.supported_concepts}

    def update(self, scores: Mapping[int, float], sim_time_ms: float) -> tuple[Event, ...]:
        now = _number(sim_time_ms, "sim_time_ms")
        if now < self._now:
            raise ProtocolError("time_reversed")
        if not isinstance(scores, Mapping) or any(type(key) is not int for key in scores) or set(scores) != set(self.supported_concepts):
            raise ProtocolError("wrong_score_concepts")
        checked = {key: _number(score, "score") for key, score in scores.items()}
        if any(score > 1 for score in checked.values()):
            raise ProtocolError("invalid_score")
        self._now = now
        if self._last_sample is not None and now - self._last_sample < self.sample_ms - 1e-9:
            return ()
        skipped = self._last_sample is not None and now - self._last_sample > self.sample_ms + 1e-9
        self._last_sample = now
        emitted = []
        for concept_id, state in self._states.items():
            score = checked[concept_id]
            state["score"] = score
            crossed = score <= self.off_threshold if state["active"] else score >= self.on_threshold
            if skipped:
                state["pending_count"] = 0
            state["pending_count"] = state["pending_count"] + 1 if crossed else 0
            transition = state["pending_count"] >= self.confirm_samples
            if transition:
                state["active"] = not state["active"]
                state["pending_count"] = 0
            last = state["last_event"]
            heartbeat = state["active"] and last is not None and now - last["sim_time_ms"] >= self.heartbeat_ms - 1e-9
            if transition or heartbeat:
                self._sequence += 1
                event = Event(SCHEMA_VERSION, self.episode_id, f"out-{self._sequence:08d}",
                              "from_brain", concept_id, now, self.event_ttl_ms,
                              state["active"], "neural_readout", score)
                state["last_event"] = event.to_dict()
                emitted.append(event)
        return tuple(emitted)

    def poll_state(self, sim_time_ms: float) -> dict[int, dict[str, Any]]:
        """Read only; stale=True never forces active=False or consumes a sample."""
        now = _number(sim_time_ms, "sim_time_ms")
        if now < self._now:
            raise ProtocolError("time_reversed")
        stale = self._last_sample is None or now >= self._last_sample + self.event_ttl_ms
        return {concept: {"active": state["active"], "score": state["score"],
                          "sim_time_ms": self._last_sample, "stale": stale,
                          "last_event": deepcopy(state["last_event"])}
                for concept, state in self._states.items()}

    def _config(self) -> dict[str, Any]:
        return {"supported_concepts": list(self.supported_concepts),
                "on_threshold": self.on_threshold, "off_threshold": self.off_threshold,
                "confirm_samples": self.confirm_samples, "sample_ms": self.sample_ms,
                "heartbeat_ms": self.heartbeat_ms, "event_ttl_ms": self.event_ttl_ms}

    def get_state(self) -> dict[str, Any]:
        return {"format": "fly_semantic.output.v1", "config": self._config(),
                "episode_id": self.episode_id, "now_ms": self._now,
                "last_sample_ms": self._last_sample, "sequence": self._sequence,
                "states": {str(key): deepcopy(value) for key, value in self._states.items()}}

    def set_state(self, value: Mapping[str, Any]) -> None:
        try:
            if set(value) != set(self.get_state()) or value["format"] != "fly_semantic.output.v1" or value["config"] != self._config():
                raise ValueError
            candidate = OutputStateMachine(value["episode_id"], **self._config())
            candidate._now = _number(value["now_ms"], "now_ms")
            last = value["last_sample_ms"]
            candidate._last_sample = None if last is None else _number(last, "last_sample_ms")
            if last is not None and last > candidate._now:
                raise ValueError
            sequence = value["sequence"]
            if type(sequence) is not int or sequence < 0:
                raise ValueError
            candidate._sequence = sequence
            if set(value["states"]) != {str(item) for item in self.supported_concepts}:
                raise ValueError
            for concept in self.supported_concepts:
                state = value["states"][str(concept)]
                if set(state) != {"active", "pending_count", "score", "last_event"} or type(state["active"]) is not bool:
                    raise ValueError
                count = state["pending_count"]
                if type(count) is not int or not 0 <= count < self.confirm_samples:
                    raise ValueError
                if (state["score"] is None) != (last is None):
                    raise ValueError
                if state["score"] is not None and _number(state["score"], "score") > 1:
                    raise ValueError
                if state["last_event"] is not None:
                    event = Event.from_dict(state["last_event"])
                    if event.episode_id != candidate.episode_id or event.concept_id != concept or event.direction != "from_brain" or event.active != state["active"] or last is None or event.sim_time_ms > last or event.ttl_ms != self.event_ttl_ms:
                        raise ValueError
                    prefix = "out-"
                    if not event.event_id.startswith(prefix) or not event.event_id[len(prefix):].isdigit() or not 1 <= int(event.event_id[len(prefix):]) <= sequence:
                        raise ValueError
                elif state["active"]:
                    raise ValueError
                candidate._states[concept] = deepcopy(state)
        except (TypeError, ValueError, KeyError, AttributeError):
            raise ProtocolError("invalid_output_checkpoint") from None
        self.__dict__.update(candidate.__dict__)
