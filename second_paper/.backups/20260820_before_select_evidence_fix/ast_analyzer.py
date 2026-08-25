"""Deterministic recovery of common Pandas intent from Python AST."""
from __future__ import annotations
import ast, re
from typing import Any, Dict, List, Optional, Tuple
from .intent_schema import CodeIntent, FilterSpec
from .grounding import closest_column, normalize_number, normalize_text, table_grounding, values_equal

_OPS={ast.Eq:"=",ast.NotEq:"!=",ast.Gt:">",ast.GtE:">=",ast.Lt:"<",ast.LtE:"<=",ast.In:"in",ast.NotIn:"not in"}

def _literal(node: ast.AST) -> Any:
    try: return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name): return node.id
        return ast.unparse(node)

def _column(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript):
        # A generated program commonly assigns filtered frames to an alias
        # (e.g. lost_games['points']). The string key is still a table column.
        key=node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value,str): return key.value
        if isinstance(key, ast.Index): return _column(key.value)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id=="df" and node.attr not in {"loc","iloc","at","iat"}: return node.attr
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
        self.columns: List[str]=[]; self.derived_columns: List[str]=[]; self.filters: List[FilterSpec]=[]; self.operations: List[str]=[]
        self.aggregation=None; self.ranking=None; self.arithmetic=None; self.evidence=[]; self.return_node=None
    def add_col(self,c):
        if c and c not in self.columns: self.columns.append(c)
    def visit_Subscript(self,node):
        c=_column(node)
        if c:
            self.add_col(c); self.evidence.append(f"column_access:{c}")
        elif isinstance(node.value, ast.Attribute) and node.value.attr in {"loc","iloc"}:
            # df.loc[mask, "target"] exposes the selected target in a tuple.
            target_nodes=[]
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                target_nodes=[node.slice.elts[-1]]
            else:
                target_nodes=[node.slice]
            for target_node in target_nodes:
                if isinstance(target_node, ast.Constant) and isinstance(target_node.value,str):
                    self.add_col(target_node.value); self.evidence.append(f"column_access:{target_node.value}")
                elif isinstance(target_node, ast.List):
                    for child in target_node.elts:
                        if isinstance(child, ast.Constant) and isinstance(child.value,str):
                            self.add_col(child.value); self.evidence.append(f"column_access:{child.value}")
        elif isinstance(node.slice, ast.Name):
            self.evidence.append("dynamic_column_access")
        self.generic_visit(node)
    def visit_Compare(self,node):
        if len(node.ops)==1 and len(node.comparators)==1:
            left_col=_find_column(node.left)
            right_col=_find_column(node.comparators[0])
            col=left_col or right_col
            op=_OPS.get(type(node.ops[0]),ast.unparse(node.ops[0]))
            if col:
                other=node.comparators[0] if left_col else node.left
                self.filters.append(FilterSpec(col,op,_literal(other))); self.operations.append("filter")
                self.evidence.append(f"filter:{col} {op} {_literal(other)!r}")
            else:
                # Scalar comparisons (including max(...) <= threshold and
                # count_a > count_b) are comparison operators, not filters.
                self.operations.append("compare")
                self.evidence.append(f"scalar_compare:{ast.unparse(node)}")
        self.generic_visit(node)
    def visit_Call(self,node):
        attr=node.func.attr if isinstance(node.func,ast.Attribute) else None
        func_name=node.func.id if isinstance(node.func,ast.Name) else None
        if func_name == "len":
            self.operations.append("aggregate"); self.aggregation="count"
            self.evidence.append("aggregation:count")
        if attr in {"unique", "nunique"}:
            self.operations.append("aggregate"); self.aggregation="count"
            self.evidence.append("aggregation:count")
        if attr:
            mapping={"sum":"aggregate","mean":"aggregate","count":"aggregate","max":"aggregate","min":"aggregate","median":"aggregate","std":"aggregate","var":"aggregate","idxmax":"ranking","idxmin":"ranking","nlargest":"ranking","nsmallest":"ranking","sort_values":"sort","rank":"ranking","groupby":"groupby","merge":"join","join":"join","any":"boolean_reduce","all":"boolean_reduce","isin":"filter","contains":"filter","value_counts":"group_by"}
            op=mapping.get(attr)
            if op: self.operations.append(op)
            if attr=="contains":
                col=_find_column(node.func.value)
                if col and node.args:
                    self.filters.append(FilterSpec(col,"contains",_literal(node.args[0])))
                    self.evidence.append(f"filter:{col} contains {_literal(node.args[0])!r}")
            if attr=="value_counts":
                self.operations.append("aggregate"); self.evidence.append("aggregation:count")
            if attr in {"sum","mean","count","max","min","median","std","var"}:
                # sum(bool_mask) is the pandas idiom for counting rows.
                self.aggregation="count" if attr=="sum" and isinstance(getattr(node.func,"value",None), ast.Compare) else attr
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
        for target in node.targets:
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=="df":
                key=target.slice
                if isinstance(key,ast.Constant) and isinstance(key.value,str):
                    if key.value not in self.derived_columns: self.derived_columns.append(key.value)
                    self.evidence.append(f"derived_column:{key.value}")
        self.generic_visit(node)

def _filter_mask(df, spec):
    column=closest_column(spec.column,df.columns)
    if column is None:
        return None
    series=df[column]
    if spec.op=="contains":
        target=normalize_text(spec.value)
        return series.map(lambda value: target in normalize_text(value))
    if spec.op=="=":
        return series.map(lambda value: values_equal(value,spec.value))
    if spec.op=="!=":
        return series.map(lambda value: not values_equal(value,spec.value))
    right=normalize_number(spec.value)
    if right is None:
        return None
    numeric=series.map(normalize_number)
    return {">":numeric>right,">=":numeric>=right,"<":numeric<right,"<=":numeric<=right}.get(spec.op)


def analyze_code(code: str, runtime: Optional[Dict[str,Any]]=None, df: Any=None) -> CodeIntent:
    try: tree=ast.parse(code)
    except SyntaxError as exc:
        return CodeIntent(ast_evidence=[f"syntax_error:{exc}"])
    v=_Visitor(); v.visit(tree)
    selected_column = _find_column(v.return_node) if v.return_node is not None else None
    if selected_column:
        v.add_col(selected_column)
        if "select" not in v.operations:
            v.operations.append("select")
        v.evidence.append(f"select:{selected_column}")
    boolean_polarity = "negative" if re.search(r"(?:answer\s*=\s*not\b|return\s+not\b)", code, flags=re.I) else "positive"
    # Keep the ordered sequence for local operation alignment.  The distinct
    # operation set is retained for compatibility with the existing auditor.
    operation_sequence=list(v.operations)
    ops=[]
    for op in operation_sequence:
        if op not in ops: ops.append(op)
    if v.filters and "filter" not in ops:
        ops.insert(0,"filter")
        operation_sequence.insert(0,"filter")
    observed=runtime or {}; rt=observed.get("observed_output_type","unknown")
    value=observed.get("observed_value")
    if rt=="unknown":
        if v.aggregation or v.arithmetic: rt="number"
        elif v.ranking or "tolist" in code: rt="list"
    cardinality="multiple" if rt=="list" else "single"
    runtime_evidence=[]
    if isinstance(observed.get("evidence"), dict):
        runtime_evidence=list(observed["evidence"].get("operation_trace", []))
        runtime_evidence.extend({"operation":"warning","value":warning} for warning in observed["evidence"].get("semantic_warnings", []))
    grounding_evidence=[]
    if df is not None:
        grounding_evidence=table_grounding(df,v.columns,v.filters).get("evidence",[])
        rows_before=int(len(df))
        for spec in v.filters:
            mask=_filter_mask(df,spec)
            if mask is not None:
                runtime_evidence.insert(max(1,len(runtime_evidence)-1),{
                    "operation":"filter","source":"ast_runtime_fusion","column":spec.column,
                    "condition":f"{spec.op} {spec.value}","rows_before":rows_before,
                    "rows_after":int(mask.fillna(False).sum()),
                })
        if selected_column:
            runtime_evidence.insert(max(1,len(runtime_evidence)-1),{
                "operation":"select","source":"ast_runtime_fusion","column":selected_column,"value":value,
            })
    return CodeIntent(return_type=rt,cardinality=cardinality,used_columns=v.columns,derived_columns=v.derived_columns,filters=v.filters,
        operations=ops,operation_sequence=operation_sequence,aggregation=v.aggregation,ranking=v.ranking,arithmetic=v.arithmetic,boolean_polarity=boolean_polarity,
        observed_output_type=rt,observed_value=value,ast_evidence=v.evidence,runtime_evidence=runtime_evidence,grounding_evidence=grounding_evidence)
