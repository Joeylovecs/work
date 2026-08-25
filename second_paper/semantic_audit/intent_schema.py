"""Serializable intent and audit schemas for semantic logic auditing."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

@dataclass
class FilterSpec:
    column: str
    op: str
    value: Any
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class QuestionIntent:
    answer_type: str = "unknown"
    cardinality: str = "single"
    task_type: str = "unknown"
    target_columns: List[str] = field(default_factory=list)
    filters: List[FilterSpec] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    operation_sequence: List[str] = field(default_factory=list)
    aggregation: Optional[str] = None
    ranking: Optional[Dict[str, Any]] = None
    arithmetic: Optional[str] = None
    boolean_polarity: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "heuristic"
    def to_dict(self) -> Dict[str, Any]:
        d=asdict(self); d["filters"]=[x.to_dict() if isinstance(x,FilterSpec) else x for x in self.filters]; return d

@dataclass
class CodeIntent:
    return_type: str = "unknown"
    cardinality: str = "single"
    used_columns: List[str] = field(default_factory=list)
    derived_columns: List[str] = field(default_factory=list)
    filters: List[FilterSpec] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    operation_sequence: List[str] = field(default_factory=list)
    aggregation: Optional[str] = None
    ranking: Optional[Dict[str, Any]] = None
    arithmetic: Optional[str] = None
    boolean_polarity: Optional[str] = None
    observed_output_type: str = "unknown"
    observed_value: Any = None
    ast_evidence: List[str] = field(default_factory=list)
    runtime_evidence: List[str] = field(default_factory=list)
    grounding_evidence: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        d=asdict(self); d["filters"]=[x.to_dict() if isinstance(x,FilterSpec) else x for x in self.filters]; return d

@dataclass
class SemanticError:
    level: str
    error_type: str
    expected: Any = None
    actual: Any = None
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    repair_hint: str = ""
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class LevelAudit:
    level: str
    passed: bool = True
    errors: List[SemanticError] = field(default_factory=list)
    confidence: float = 1.0
    def to_dict(self) -> Dict[str, Any]:
        return {"level":self.level,"passed":self.passed,"errors":[e.to_dict() for e in self.errors],"confidence":self.confidence}

@dataclass
class AuditResult:
    passed: bool
    semantic_exception: bool
    global_audit: LevelAudit
    operator_audit: LevelAudit
    parameter_audit: LevelAudit
    repair_hint: str = ""
    confidence: float = 0.0
    mode: str = "hybrid"
    def to_dict(self) -> Dict[str, Any]:
        return {"passed":self.passed,"semantic_exception":self.semantic_exception,
                "global":self.global_audit.to_dict(),"operator":self.operator_audit.to_dict(),
                "parameter":self.parameter_audit.to_dict(),"repair_hint":self.repair_hint,
                "confidence":self.confidence,"mode":self.mode}

@dataclass
class SemanticLogicException(Exception):
    level: str
    error_type: str
    expected: Any = None
    actual: Any = None
    evidence: List[str] = field(default_factory=list)
    repair_hint: str = ""
    confidence: float = 0.0
    def __str__(self) -> str:
        return f"SemanticLogicException[{self.level}/{self.error_type}]: {self.repair_hint}"

def coerce_filter(value: Any) -> FilterSpec:
    if isinstance(value, FilterSpec): return value
    if isinstance(value, dict): return FilterSpec(str(value.get("column","")), str(value.get("op","=")), value.get("value"))
    raise TypeError(f"Unsupported filter: {value!r}")

def question_from_dict(data: Dict[str, Any]) -> QuestionIntent:
    d=dict(data); d["filters"]=[coerce_filter(x) for x in d.get("filters",[])]; return QuestionIntent(**{k:v for k,v in d.items() if k in QuestionIntent.__dataclass_fields__})

def code_from_dict(data: Dict[str, Any]) -> CodeIntent:
    d=dict(data); d["filters"]=[coerce_filter(x) for x in d.get("filters",[])]; return CodeIntent(**{k:v for k,v in d.items() if k in CodeIntent.__dataclass_fields__})

def dumps(value: Any) -> str:
    if hasattr(value,"to_dict"): value=value.to_dict()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
