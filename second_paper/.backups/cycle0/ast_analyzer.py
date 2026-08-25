"""Deterministic recovery of common Pandas intent from Python AST."""
from __future__ import annotations
import ast, re
from typing import Any, Dict, List, Optional, Tuple
from .intent_schema import CodeIntent, FilterSpec

_OPS={ast.Eq:"=",ast.NotEq:"!=",ast.Gt:">",ast.GtE:">=",ast.Lt:"<",ast.LtE:"<=",ast.In:"in",ast.NotIn:"not in"}

def _literal(node: ast.AST) -> Any:
    try: return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name): return node.id
        return ast.unparse(node)

def _column(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript):
        base=node.value
        if isinstance(base, ast.Name) and base.id in {"df","filtered","result","subset","data"}:
            key=node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value,str): return key.value
            if isinstance(key, ast.Index): return _column(key.value)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id=="df": return node.attr
    return None

def _find_column(node: ast.AST) -> Optional[str]:
    c=_column(node)
    if c: return c
    for child in ast.walk(node):
        c=_column(child)
        if c: return c
    return None

class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.columns: List[str]=[]; self.filters: List[FilterSpec]=[]; self.operations: List[str]=[]
        self.aggregation=None; self.ranking=None; self.arithmetic=None; self.evidence=[]; self.return_node=None
    def add_col(self,c):
        if c and c not in self.columns: self.columns.append(c)
    def visit_Subscript(self,node):
        c=_column(node)
        if c: self.add_col(c); self.evidence.append(f"column_access:{c}")
        self.generic_visit(node)
    def visit_Compare(self,node):
        if len(node.ops)==1 and len(node.comparators)==1:
            col=_find_column(node.left) or _find_column(node.comparators[0])
            if col:
                op=_OPS.get(type(node.ops[0]),ast.unparse(node.ops[0]))
                other=node.comparators[0] if _find_column(node.left) else node.left
                self.filters.append(FilterSpec(col,op,_literal(other))); self.operations.append("filter")
                self.evidence.append(f"filter:{col} {op} {_literal(other)!r}")
        self.generic_visit(node)
    def visit_Call(self,node):
        attr=node.func.attr if isinstance(node.func,ast.Attribute) else None
        if attr:
            mapping={"sum":"aggregate","mean":"aggregate","count":"aggregate","max":"aggregate","min":"aggregate","median":"aggregate","std":"aggregate","var":"aggregate","idxmax":"ranking","idxmin":"ranking","nlargest":"ranking","nsmallest":"ranking","sort_values":"sort","rank":"ranking","groupby":"groupby","merge":"join","join":"join","any":"boolean_reduce","all":"boolean_reduce","isin":"filter"}
            op=mapping.get(attr)
            if op: self.operations.append(op)
            if attr in {"sum","mean","count","max","min","median","std","var"}: self.aggregation=attr
            if attr in {"idxmax","nlargest","sort_values"}: self.ranking={"direction":"desc","operation":attr}
            if attr in {"idxmin","nsmallest"}: self.ranking={"direction":"asc","operation":attr}
            if attr=="sort_values" and len(node.args)>=1: self.ranking={"direction":"desc","operation":attr,"column":_literal(node.args[0])}
            if attr=="groupby": self.evidence.append("groupby")
            if attr in {"sum","mean","count","max","min"}: self.evidence.append(f"aggregation:{attr}")
            if attr in {"idxmax","idxmin","nlargest","nsmallest","sort_values"}: self.evidence.append(f"ranking:{attr}")
        self.generic_visit(node)
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Sub): self.arithmetic="difference"; self.operations.append("arithmetic")
        elif isinstance(node.op,ast.Add): self.arithmetic="addition"; self.operations.append("arithmetic")
        elif isinstance(node.op,ast.Mult): self.arithmetic="multiplication"; self.operations.append("arithmetic")
        elif isinstance(node.op,ast.Div): self.arithmetic="division"; self.operations.append("arithmetic")
        self.generic_visit(node)
    def visit_Return(self,node): self.return_node=node.value; self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in {"answer","result","output","prediction"} for t in node.targets): self.return_node=node.value
        self.generic_visit(node)

def analyze_code(code: str, runtime: Optional[Dict[str,Any]]=None) -> CodeIntent:
    try: tree=ast.parse(code)
    except SyntaxError as exc:
        return CodeIntent(ast_evidence=[f"syntax_error:{exc}"])
    v=_Visitor(); v.visit(tree)
    boolean_polarity = "false" if re.search(r"\bnot\b", code) else "true"
    ops=[]
    for op in v.operations:
        if op not in ops: ops.append(op)
    if v.filters and "filter" not in ops: ops.insert(0,"filter")
    observed=runtime or {}; rt=observed.get("observed_output_type","unknown")
    value=observed.get("observed_value")
    if rt=="unknown":
        if v.aggregation or v.arithmetic: rt="number"
        elif v.ranking or "tolist" in code: rt="list"
    cardinality="multiple" if rt=="list" else "single"
    return CodeIntent(return_type=rt,cardinality=cardinality,used_columns=v.columns,filters=v.filters,
        operations=ops,aggregation=v.aggregation,ranking=v.ranking,arithmetic=v.arithmetic,boolean_polarity=boolean_polarity,
        observed_output_type=rt,observed_value=value,ast_evidence=v.evidence)
