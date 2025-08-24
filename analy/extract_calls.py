#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_calls.py — Static extractor for functions, classes, imports and call graph.

Features
- List all functions (module-level & methods) with signature and location.
- List all classes and methods.
- List imports (import & from-import).
- Build a lightweight call graph: for each function/method, which call expressions it contains.
- Try to infer the "execution order" under `if __name__ == "__main__":` and top-level calls.
- Optionally emit JSON and Graphviz .dot.

Limitations
- Static heuristics only; no dynamic dispatch resolution.
- Attribute calls are recorded as dotted names if possible (e.g., module.func, obj.method).
- Does not execute code.

Usage:
  python extract_calls.py path/to/mainpopperarc.py [--json] [--dot out.dot]
  python extract_calls.py path/to/dir --recursive [--follow-local-imports]

Author: ChatGPT (generated)
"""

import ast
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union

@dataclass
class FuncInfo:
    qualname: str
    name: str
    kind: str  # "function" or "method"
    lineno: int
    endlineno: Optional[int]
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)

@dataclass
class ClassInfo:
    name: str
    lineno: int
    endlineno: Optional[int]
    bases: List[str] = field(default_factory=list)
    methods: List[FuncInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None

@dataclass
class ModuleReport:
    path: str
    module: str
    functions: List[FuncInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    main_calls: List[str] = field(default_factory=list)   # calls in if __name__ == "__main__"
    toplevel_calls: List[str] = field(default_factory=list)

def get_name(node: ast.AST) -> str:
    """Return a dotted name for Name/Attribute nodes, else a generic tag."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{get_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return get_name(node.func)
    if isinstance(node, ast.Subscript):
        return get_name(node.value)
    if isinstance(node, ast.Lambda):
        return "<lambda>"
    return node.__class__.__name__

class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call):
        name = get_name(node.func)
        self.calls.append(name)
        self.generic_visit(node)

class ModuleVisitor(ast.NodeVisitor):
    def __init__(self, module_path: Path):
        self.module_path = module_path
        self.report = ModuleReport(path=str(module_path), module=module_path.stem)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.report.imports.append(alias.name if alias.asname is None else f"{alias.name} as {alias.asname}")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        level = node.level or 0
        prefix = "." * level + mod
        for alias in node.names:
            self.report.from_imports.append(f"from {prefix} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))

    def _collect_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], qual_prefix: str, kind: str) -> FuncInfo:
        args = []
        for a in node.args.args:
            args.append(a.arg)
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        for a in node.args.kwonlyargs:
            args.append(a.arg)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)

        decorators = [get_name(d) for d in node.decorator_list]
        returns = None
        if node.returns:
            returns = get_name(node.returns)

        doc = ast.get_docstring(node)
        qual = f"{qual_prefix}.{node.name}" if qual_prefix else node.name

        # collect calls
        cc = CallCollector()
        for b in node.body:
            cc.visit(b)

        return FuncInfo(
            qualname=qual,
            name=node.name,
            kind=kind,
            lineno=node.lineno,
            endlineno=getattr(node, "end_lineno", None),
            args=args,
            returns=returns,
            decorators=decorators,
            docstring=doc,
            calls=cc.calls
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        fi = self._collect_function(node, "", "function")
        self.report.functions.append(fi)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        fi = self._collect_function(node, "", "function")
        self.report.functions.append(fi)

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = [get_name(b) for b in node.bases]
        decorators = [get_name(d) for d in node.decorator_list]
        doc = ast.get_docstring(node)
        ci = ClassInfo(name=node.name, lineno=node.lineno, endlineno=getattr(node, "end_lineno", None),
                       bases=bases, decorators=decorators, docstring=doc)

        # methods
        for b in node.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mi = self._collect_function(b, node.name, "method")
                ci.methods.append(mi)
        self.report.classes.append(ci)

    def visit_If(self, node: ast.If):
        # detect if __name__ == "__main__"
        def is_main_check(n):
            if isinstance(n, ast.Compare):
                if isinstance(n.left, ast.Name) and n.left.id == "__name__":
                    for op in n.ops:
                        if not isinstance(op, (ast.Eq, ast.Is)):
                            return False
                    for c in n.comparators:
                        if isinstance(c, ast.Constant) and c.value == "__main__":
                            return True
            return False

        if is_main_check(node.test):
            # collect call order in this block
            calls = []
            for b in node.body:
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Call):
                    calls.append(get_name(b.value.func))
                else:
                    # also dive to find nested calls
                    cc = CallCollector()
                    cc.visit(b)
                    calls.extend(cc.calls)
            self.report.main_calls.extend(calls)

        # still traverse inside
        self.generic_visit(node)

def analyze_file(path: Path) -> ModuleReport:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    v = ModuleVisitor(path)
    v.visit(tree)

    # toplevel calls (outside of functions/classes)
    toplevel_calls: List[str] = []
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            toplevel_calls.append(get_name(n.value.func))
    v.report.toplevel_calls = toplevel_calls
    return v.report

def emit_markdown(rep: ModuleReport) -> str:
    lines = []
    lines.append(f"# Report for `{rep.path}` (module `{rep.module}`)\n")
    if rep.imports or rep.from_imports:
        lines.append("## Imports")
        if rep.imports:
            for i in rep.imports:
                lines.append(f"- `import {i}`")
        if rep.from_imports:
            for i in rep.from_imports:
                lines.append(f"- `{i}`")
        lines.append("")

    if rep.functions:
        lines.append("## Functions")
        for f in sorted(rep.functions, key=lambda x: x.lineno):
            sig = f"({', '.join(f.args)})"
            lines.append(f"- **{f.qualname}{sig}**  ⟶  L{f.lineno}" + (f"-L{f.endlineno}" if f.endlineno else ""))
            if f.returns:
                lines.append(f"  - returns: `{f.returns}`")
            if f.decorators:
                lines.append(f"  - decorators: {', '.join(f.decorators)}")
            if f.docstring:
                ds = f.docstring.strip().splitlines()[0]
                lines.append(f"  - doc: {ds}")
            if f.calls:
                preview = ", ".join(f.calls[:10])
                more = "" if len(f.calls) <= 10 else f" … (+{len(f.calls)-10} more)"
                lines.append(f"  - calls: {preview}{more}")
        lines.append("")

    if rep.classes:
        lines.append("## Classes")
        for c in sorted(rep.classes, key=lambda x: x.lineno):
            base = f" : {', '.join(c.bases)}" if c.bases else ""
            lines.append(f"- **class {c.name}{base}**  ⟶  L{c.lineno}" + (f"-L{c.endlineno}" if c.endlineno else ""))
            if c.decorators:
                lines.append(f"  - decorators: {', '.join(c.decorators)}")
            if c.docstring:
                ds = c.docstring.strip().splitlines()[0]
                lines.append(f"  - doc: {ds}")
            if c.methods:
                lines.append(f"  - methods:")
                for m in sorted(c.methods, key=lambda x: x.lineno):
                    sig = f"({', '.join(m.args)})"
                    lines.append(f"    - **{m.qualname}{sig}**  ⟶  L{m.lineno}" + (f"-L{m.endlineno}" if m.endlineno else ""))
                    if m.calls:
                        preview = ", ".join(m.calls[:10])
                        more = "" if len(m.calls) <= 10 else f" … (+{len(m.calls)-10} more)"
                        lines.append(f"      - calls: {preview}{more}")
        lines.append("")

    if rep.main_calls or rep.toplevel_calls:
        lines.append("## Entry / Execution Order (static)")
        if rep.main_calls:
            lines.append("- Calls inside `if __name__ == \"__main__\"`: ")
            lines.append("  1. " + "\n  1. ".join(rep.main_calls))
        if rep.toplevel_calls:
            lines.append("- Top-level calls: " + ", ".join(rep.toplevel_calls))
        lines.append("")

    lines.append("## Notes")
    lines.append("- This is static analysis. Dynamic dispatch, higher-order calls, and reflection will not be fully resolved.")
    return "\n".join(lines)

def build_dot(rep: ModuleReport) -> str:
    # Create a simple call graph (functions/methods -> called names)
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()

    def add_edges(owner: str, calls: List[str]):
        for callee in calls:
            nodes.add(callee)
            edges.add((owner, callee))

    for f in rep.functions:
        nodes.add(f.qualname); add_edges(f.qualname, f.calls)
    for c in rep.classes:
        for m in c.methods:
            nodes.add(m.qualname); add_edges(m.qualname, m.calls)

    # also link __main__ pseudo-node
    if rep.main_calls:
        nodes.add("__main__")
        for callee in rep.main_calls:
            edges.add(("__main__", callee))

    # DOT
    out = ["digraph G {", "  rankdir=LR;"]
    for n in sorted(nodes):
        out.append(f'  "{n}" [shape=box];')
    for a,b in sorted(edges):
        out.append(f'  "{a}" -> "{b}";')
    out.append("}")
    return "\n".join(out)

def walk_inputs(root: Path, recursive: bool=False) -> List[Path]:
    if root.is_file():
        return [root]
    if not recursive:
        return [p for p in root.iterdir() if p.suffix == ".py"]
    paths = []
    for p in root.rglob("*.py"):
        paths.append(p)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Python file or directory")
    ap.add_argument("--recursive", action="store_true", help="When path is a dir, recurse into subdirs")
    ap.add_argument("--json", action="store_true", help="Emit a JSON file next to the input")
    ap.add_argument("--dot", help="Emit Graphviz .dot to given path (or <input>.dot if path is '-')", default=None)
    args = ap.parse_args()

    inputs = walk_inputs(Path(args.path), recursive=args.recursive)
    for in_path in inputs:
        rep = analyze_file(in_path)
        md = emit_markdown(rep)
        md_path = in_path.with_suffix(in_path.suffix + ".report.md")
        md_path.write_text(md, encoding="utf-8")
        print(f"[OK] Markdown report: {md_path}")

        if args.json:
            # Convert dataclasses to serializable dicts
            def f2d(f: FuncInfo): return {
                **{k:getattr(f,k) for k in ("qualname","name","kind","lineno","endlineno","args","returns","decorators","docstring","calls")}
            }
            def c2d(c: ClassInfo): return {
                "name": c.name, "lineno": c.lineno, "endlineno": c.endlineno, "bases": c.bases,
                "decorators": c.decorators, "docstring": c.docstring, "methods": [f2d(m) for m in c.methods]
            }
            out = {
                "path": rep.path, "module": rep.module,
                "functions": [f2d(f) for f in rep.functions],
                "classes": [c2d(c) for c in rep.classes],
                "imports": rep.imports, "from_imports": rep.from_imports,
                "main_calls": rep.main_calls, "toplevel_calls": rep.toplevel_calls,
            }
            json_path = in_path.with_suffix(in_path.suffix + ".calls.json")
            json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] JSON: {json_path}")

        if args.dot is not None:
            dot_text = build_dot(rep)
            if args.dot == "-":
                dot_path = in_path.with_suffix(in_path.suffix + ".dot")
            else:
                dot_path = Path(args.dot)
            dot_path.write_text(dot_text, encoding="utf-8")
            print(f"[OK] DOT: {dot_path}")

if __name__ == "__main__":
    main()
