from dataclasses import dataclass, field
from typing import Dict, List, Set, Union, Tuple


@dataclass
class Intent:
    name: str
    description: str
    required_slots: List[str]
    optional_slots: List[str]
    pr_name: str
    pr_name_description: str


@dataclass
class Slot:
    name: str
    description: str
    is_categorical: bool
    possible_values: List[str]
    pr_name: str
    pr_name_description: str


@dataclass
class Service:
    name: str
    description: str
    slots: Dict[str, Slot]
    intents: Dict[str, Intent]
    required_slots: Set[str]
    optional_slots: Set[str]
    usr_slots: List[str]
    pr_name: str


@dataclass
class Schema:
    services: Dict[str, Service]


@dataclass
class Action:
    name: str
    slot: Union[str, None]
    values: List[str]


@dataclass
class State:
    active_intent: str
    req_slots: List[str]
    sv_pairs: Dict[str, List[str]]


@dataclass
class Frame:
    actions: List[Action]
    state: Union[State, None]
    slot_spans: Dict[str, Tuple[int, int]]


@dataclass
class Turn:
    id: int
    speaker: str
    utterance: str
    frames: Dict[str, Frame]


@dataclass
class ServiceHistory:
    latest_sys_sv_pairs: Dict[str, List[str]] = field(default_factory=dict)
    sys_sv_pairs: Dict[str, List[str]] = field(default_factory=dict)
    prev_sys_sv_pairs: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DialogueHistory:
    services_so_far: Set[str] = field(default_factory=set)
    prev_turn_services: Set[str] = field(default_factory=set)
    prev_sys_uttr: str = ''
    services: Dict[str, ServiceHistory] = field(default_factory=dict)

    def is_multi_domain(self):
        return len(self.services) > 1


@dataclass
class ServiceHistoryPred:
    usr_sv_pairs: Dict[str, List[str]] = field(default_factory=dict)
    intent: str = 'NONE'


@dataclass
class DialogueHistoryPred:
    services: Dict[str, ServiceHistoryPred] = field(default_factory=dict)


@dataclass
class Dialogue:
    services: List[str]
    turns: Dict[int, Turn]
