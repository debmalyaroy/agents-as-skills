//! Parallel `tree-sitter-python` AST indexer, call and inheritance graph (including `async def`),
//! mean-centred 2-hop structural embeddings and the AST analyzers of the static critic.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

pub const EMBED_DIM: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Signature {
    pub pos: Vec<String>,
    pub req: usize,
    pub kwonly: Vec<String>,
    pub kwreq: Vec<String>,
    pub var: bool,
    pub kw: bool,
    pub method: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefNode {
    /// Qualified node id (`module.Class.method` or `module.func`).
    pub id: String,
    /// Bare name (`method` or `func`).
    pub name: String,
    /// File-local qualified name (`Class.method` or `func`).
    pub qualname: String,
    /// Repo-relative path using forward slashes (`pkg/mod.py`).
    pub file: String,
    /// `"function"`, `"async_function"`, or `"class"`.
    pub kind: String,
    /// 1-indexed start line (including decorators if decorated).
    pub start_line: usize,
    /// 1-indexed line of the `def` / `async def` / `class` keyword.
    pub def_line: usize,
    /// 1-indexed first executable body line after any docstring.
    pub body_start_line: usize,
    /// 1-indexed inclusive end line.
    pub end_line: usize,
    /// Parameter signature (for functions/methods).
    pub sig: Option<Signature>,
    /// Extracted calls `(receiver, func_name)` inside the definition.
    pub calls: Vec<(Option<String>, String)>,
    /// Base class names (for classes).
    pub bases: Vec<String>,
    /// Structural + identifier sub-tokens for BM25 and embedding construction.
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepoIndex {
    /// All indexed definitions keyed by `id`.
    pub nodes: BTreeMap<String, DefNode>,
    /// File -> ordered list of `DefNode.id`s in that file.
    pub file_defs: BTreeMap<String, Vec<String>>,
    /// Undirected adjacency list (`node_id -> neighbor_ids`).
    pub adj: BTreeMap<String, Vec<String>>,
    /// Directed callers (`callee_id -> caller_ids`).
    pub callers: BTreeMap<String, Vec<String>>,
    /// Directed callees (`caller_id -> callee_ids`).
    pub callees: BTreeMap<String, Vec<String>>,
    /// Directed subclasses (`base_id -> subclass_ids`, up to 2 hops).
    pub subclasses: BTreeMap<String, Vec<String>>,
    /// Directed superclasses (`subclass_id -> base_ids`, up to 2 hops).
    pub superclasses: BTreeMap<String, Vec<String>>,
    /// Paired test files (`node_id -> test_files`).
    pub tests_of: BTreeMap<String, Vec<String>>,
    /// Mean-centred 256-d structural embeddings keyed by `node_id` (non-test nodes).
    pub embeddings: BTreeMap<String, Vec<f32>>,
    /// All tracked `.py` files in the repository.
    pub files: Vec<String>,
}

/// Convert a repo-relative `.py` file path to a dotted module name (`pkg.mod`) using `Path::components`.
pub fn module_name(path: &str) -> String {
    let no_ext = Path::new(path).with_extension("");
    let mut parts: Vec<String> = no_ext
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(os) => os.to_str().map(String::from),
            _ => None,
        })
        .collect();
    if parts.len() > 1 && (parts[0] == "src" || parts[0] == "lib" || parts[0].ends_with("src")) {
        parts.remove(0);
    }
    if parts.last().map(String::as_str) == Some("__init__") {
        parts.pop();
    }
    parts.join(".")
}

pub fn is_test_path(path: &str) -> bool {
    let segs: Vec<String> = crate::path_segments(path)
        .into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    segs.iter().any(|s| s == "tests" || s == "test")
        || segs.last().is_some_and(|b| {
            b.starts_with("test_") || b.ends_with("_test.py") || b == "conftest.py"
        })
}

pub fn is_docs_path(path: &str) -> bool {
    let segs: Vec<String> = crate::path_segments(path)
        .into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    let first = segs.first().map(String::as_str).unwrap_or("");
    let is_root_build_hook = segs.len() == 1
        && matches!(
            first,
            "setup.py"
                | "noxfile.py"
                | "fabfile.py"
                | "dodo.py"
                | "_griffe_ext.py"
                | "_mkdocs_hooks.py"
        );
    is_root_build_hook
        || matches!(
            first,
            "docs" | "docs_src" | "doc" | "examples" | "scripts" | "tools" | "bin" | "benchmarks"
        )
        || segs.iter().any(|s| s == "docs")
}

fn node_text<'a>(n: Node, src: &'a [u8]) -> &'a str {
    n.utf8_text(src).unwrap_or("")
}

/// Split identifier/code text into lowercase snake_case and CamelCase tokens (length >= 2).
pub fn tokenize_ident_and_subwords(s: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "self", "cls", "def", "class", "return", "none", "true", "false", "import", "from", "the",
        "and", "for", "with", "not", "this", "that", "args", "kwargs", "pass", "async", "await",
    ];
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            push_word_and_subwords(&cur, STOP, &mut out);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        push_word_and_subwords(&cur, STOP, &mut out);
    }
    out
}

fn push_word_and_subwords(word: &str, stop: &[&str], out: &mut Vec<String>) {
    if word.len() < 2 || word.chars().all(|c| c.is_ascii_digit()) {
        return;
    }
    let lw = word.to_ascii_lowercase();
    if !stop.contains(&lw.as_str()) {
        out.push(lw.clone());
    }
    // Split on '_' and CamelCase transitions
    let mut sub = String::new();
    let chars: Vec<char> = word.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            if sub.len() > 2 {
                let ls = sub.to_ascii_lowercase();
                if ls != lw && !stop.contains(&ls.as_str()) {
                    out.push(ls);
                }
            }
            sub.clear();
            continue;
        }
        let camel_split = c.is_ascii_uppercase()
            && i > 0
            && (chars[i - 1].is_ascii_lowercase()
                || (chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase())
                    && chars[i - 1].is_ascii_uppercase()));
        if camel_split && sub.len() > 2 {
            let ls = sub.to_ascii_lowercase();
            if ls != lw && !stop.contains(&ls.as_str()) {
                out.push(ls);
            }
            sub.clear();
        }
        if c.is_ascii_alphanumeric() {
            sub.push(c);
        }
    }
    if sub.len() > 2 {
        let ls = sub.to_ascii_lowercase();
        if ls != lw && !stop.contains(&ls.as_str()) {
            out.push(ls);
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Extract function signature from a `tree-sitter-python` `function_definition` node.
fn parse_signature(func_node: Node, src: &[u8], method: bool) -> Signature {
    let mut pos = Vec::new();
    let mut defaults_count = 0usize;
    let mut kwonly = Vec::new();
    let mut kwreq = Vec::new();
    let mut var = false;
    let mut kw = false;
    let mut after_star = false;

    if let Some(params) = func_node.child_by_field_name("parameters") {
        let mut cur = params.walk();
        for c in params.named_children(&mut cur) {
            match c.kind() {
                "identifier" => {
                    let name = node_text(c, src).to_string();
                    if after_star {
                        kwonly.push(name.clone());
                        kwreq.push(name);
                    } else {
                        pos.push(name);
                    }
                }
                "typed_parameter" => {
                    if let Some(first) = c.named_child(0) {
                        match first.kind() {
                            "identifier" => {
                                let name = node_text(first, src).to_string();
                                if after_star {
                                    kwonly.push(name.clone());
                                    kwreq.push(name);
                                } else {
                                    pos.push(name);
                                }
                            }
                            "list_splat_pattern" => {
                                var = true;
                                after_star = true;
                            }
                            "dictionary_splat_pattern" => {
                                kw = true;
                            }
                            _ => {}
                        }
                    }
                }
                "default_parameter" | "typed_default_parameter" => {
                    if let Some(name_node) = c.child_by_field_name("name") {
                        let name = node_text(name_node, src).to_string();
                        if after_star {
                            kwonly.push(name);
                        } else {
                            pos.push(name);
                            defaults_count += 1;
                        }
                    }
                }
                "list_splat_pattern" | "keyword_separator" => {
                    if c.kind() == "list_splat_pattern" {
                        var = true;
                    }
                    after_star = true;
                }
                "dictionary_splat_pattern" => {
                    kw = true;
                }
                "positional_separator" => {}
                _ => {}
            }
        }
    }
    let req = pos.len().saturating_sub(defaults_count);
    Signature {
        pos,
        req,
        kwonly,
        kwreq,
        var,
        kw,
        method,
    }
}

fn collect_calls(n: Node, src: &[u8], out: &mut Vec<(Option<String>, String)>) {
    if n.kind() == "call"
        && let Some(f) = n.child_by_field_name("function")
    {
        match f.kind() {
            "identifier" => out.push((None, node_text(f, src).to_string())),
            "attribute" => {
                if let (Some(obj), Some(attr)) = (
                    f.child_by_field_name("object"),
                    f.child_by_field_name("attribute"),
                ) {
                    let recv = node_text(obj, src);
                    let mname = node_text(attr, src).to_string();
                    if recv == "self" || recv == "cls" {
                        out.push((Some("self".to_string()), mname));
                    } else if obj.kind() == "identifier" {
                        out.push((Some(recv.to_string()), mname));
                    } else {
                        out.push((None, mname));
                    }
                }
            }
            _ => {}
        }
    }
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        collect_calls(c, src, out);
    }
}

fn collect_ast_shape_and_tokens(n: Node, _src: &[u8], out: &mut Vec<String>) {
    match n.kind() {
        "if_statement" | "for_statement" | "while_statement" | "try_statement"
        | "with_statement" | "return_statement" | "raise_statement" | "yield" | "await"
        | "assert_statement" | "match_statement" => {
            out.push(format!("ast:{}", n.kind()));
        }
        _ => {}
    }
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        collect_ast_shape_and_tokens(c, _src, out);
    }
}

fn first_body_line_after_docstring(def_node: Node, src: &[u8], fallback: usize) -> usize {
    let Some(body) = def_node.child_by_field_name("body") else {
        return fallback;
    };
    let mut cur = body.walk();
    let children: Vec<Node> = body.named_children(&mut cur).collect();
    if children.is_empty() {
        return fallback;
    }
    if children[0].kind() == "expression_statement"
        && let Some(inner) = children[0].named_child(0)
        && inner.kind() == "string"
        && children.len() > 1
    {
        let _ = src;
        return children[1].start_position().row + 1;
    }
    children[0].start_position().row + 1
}

/// Parse one Python file into its definitions (`def`, `async def`, `class`) and local imports.
/// Automatically collapses `@overload` type stubs into their concrete implementation `DefNode`
/// (merging `@overload` type-annotation tokens & calls while keeping the concrete `start_line..end_line`).
pub fn parse_python_file(rel_path: &str, src_str: &str) -> (Vec<DefNode>, HashMap<String, String>) {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("tree-sitter-python language");
    let src = src_str.as_bytes();
    let Some(tree) = parser.parse(src, None) else {
        return (Vec::new(), HashMap::new());
    };
    let root = tree.root_node();
    let mod_name = module_name(rel_path);
    let is_pkg = rel_path.ends_with("__init__.py");
    let imports = extract_imports(root, src, &mod_name, is_pkg);
    let src_lines: Vec<&str> = src_str.lines().collect();
    let mut raw_defs: Vec<(DefNode, bool)> = Vec::new();
    walk_defs(
        root,
        src,
        &src_lines,
        rel_path,
        &mod_name,
        "",
        &mut raw_defs,
    );

    let concrete_quals: HashSet<String> = raw_defs
        .iter()
        .filter(|(_, is_ov)| !*is_ov)
        .map(|(d, _)| d.qualname.clone())
        .collect();
    type OverloadExtra = (Vec<String>, Vec<(Option<String>, String)>);
    let mut overload_extras: HashMap<String, OverloadExtra> = HashMap::new();
    for (d, is_ov) in &raw_defs {
        if *is_ov && concrete_quals.contains(&d.qualname) {
            let entry = overload_extras.entry(d.qualname.clone()).or_default();
            if entry.0.len() < 240 {
                for tok in &d.tokens {
                    if !tok.starts_with("ast:") && entry.0.len() < 240 {
                        entry.0.push(tok.clone());
                    }
                }
            }
            for c in &d.calls {
                if !entry.1.contains(c) {
                    entry.1.push(c.clone());
                }
            }
        }
    }

    let mut defs = Vec::with_capacity(raw_defs.len());
    for (mut d, is_ov) in raw_defs {
        if is_ov && concrete_quals.contains(&d.qualname) {
            continue;
        }
        if let Some((extra_toks, extra_calls)) = overload_extras.remove(&d.qualname) {
            d.tokens.extend(extra_toks);
            for c in extra_calls {
                if !d.calls.contains(&c) {
                    d.calls.push(c);
                }
            }
        }
        defs.push(d);
    }
    (defs, imports)
}

fn extract_imports(root: Node, src: &[u8], module: &str, is_pkg: bool) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut pkg: Vec<&str> = module.split('.').filter(|p| !p.is_empty()).collect();
    if !is_pkg {
        pkg.pop();
    }
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "import_statement" => {
                let mut cur = n.walk();
                for c in n.named_children(&mut cur) {
                    if c.kind() == "dotted_name" {
                        let d = node_text(c, src);
                        let local = d.split('.').next().unwrap_or(d);
                        out.insert(local.to_string(), d.to_string());
                    } else if c.kind() == "aliased_import"
                        && let (Some(name), Some(alias)) = (
                            c.child_by_field_name("name"),
                            c.child_by_field_name("alias"),
                        )
                    {
                        out.insert(
                            node_text(alias, src).to_string(),
                            node_text(name, src).to_string(),
                        );
                    }
                }
            }
            "import_from_statement" => {
                let mut dots = 0usize;
                let mut mod_part = String::new();
                if let Some(mn) = n.child_by_field_name("module_name") {
                    let raw = node_text(mn, src);
                    dots = raw.chars().take_while(|&c| c == '.').count();
                    mod_part = raw[dots..].to_string();
                }
                let base = if dots > 0 {
                    let keep = pkg.len().saturating_sub(dots - 1);
                    let mut p: Vec<&str> = pkg[..keep].to_vec();
                    if !mod_part.is_empty() {
                        p.push(&mod_part);
                    }
                    p.join(".")
                } else {
                    mod_part
                };
                let mut cur = n.walk();
                for c in n.children_by_field_name("name", &mut cur) {
                    if c.kind() == "dotted_name" {
                        let nm = node_text(c, src);
                        let full = if base.is_empty() {
                            nm.to_string()
                        } else {
                            format!("{base}.{nm}")
                        };
                        out.insert(nm.to_string(), full);
                    } else if c.kind() == "aliased_import"
                        && let (Some(name), Some(alias)) = (
                            c.child_by_field_name("name"),
                            c.child_by_field_name("alias"),
                        )
                    {
                        let nm = node_text(name, src);
                        let full = if base.is_empty() {
                            nm.to_string()
                        } else {
                            format!("{base}.{nm}")
                        };
                        out.insert(node_text(alias, src).to_string(), full);
                    }
                }
            }
            _ => {
                let mut cur = n.walk();
                for c in n.named_children(&mut cur) {
                    stack.push(c);
                }
            }
        }
    }
    out
}

fn has_overload_decorator(outer: Node, src: &[u8]) -> bool {
    if outer.kind() != "decorated_definition" {
        return false;
    }
    let mut cur = outer.walk();
    for c in outer.named_children(&mut cur) {
        if c.kind() == "decorator" {
            let txt = node_text(c, src).trim().trim_start_matches('@').trim();
            let head = txt.split('(').next().unwrap_or(txt).trim();
            if head.rsplit('.').next() == Some("overload") {
                return true;
            }
        }
    }
    false
}

fn extract_body_aware_snippet(
    outer: Node,
    src: &[u8],
    src_lines: &[&str],
    start_line: usize,
    body_start_line: usize,
    end_line: usize,
    max_chars: usize,
) -> String {
    if body_start_line > start_line + 12 && !src_lines.is_empty() {
        let h_start = start_line.saturating_sub(1).min(src_lines.len());
        let h_end = (start_line + 8)
            .min(body_start_line.saturating_sub(1))
            .min(src_lines.len());
        let b_start = body_start_line.saturating_sub(1).min(src_lines.len());
        let b_end = end_line.min(src_lines.len());
        let head: String = src_lines[h_start..h_end]
            .join("\n")
            .chars()
            .take(700)
            .collect();
        let body: String = if b_start < b_end {
            src_lines[b_start..b_end]
                .join("\n")
                .chars()
                .take(max_chars)
                .collect()
        } else {
            String::new()
        };
        format!("{head}\n{body}")
    } else {
        node_text(outer, src).chars().take(max_chars).collect()
    }
}

fn walk_defs(
    node: Node,
    src: &[u8],
    src_lines: &[&str],
    rel_path: &str,
    mod_name: &str,
    class_prefix: &str,
    out: &mut Vec<(DefNode, bool)>,
) {
    let mut cur = node.walk();
    for child in node.named_children(&mut cur) {
        let (outer, def_node) = if child.kind() == "decorated_definition" {
            match child.child_by_field_name("definition") {
                Some(d) => (child, d),
                None => continue,
            }
        } else {
            (child, child)
        };
        match def_node.kind() {
            "function_definition" => {
                let Some(name_node) = def_node.child_by_field_name("name") else {
                    continue;
                };
                let name = node_text(name_node, src).to_string();
                let is_overload = has_overload_decorator(outer, src);
                let is_async_fn = def_node
                    .child(0)
                    .is_some_and(|c| c.kind() == "async" || node_text(c, src) == "async")
                    || node_text(def_node, src).trim_start().starts_with("async ");
                let qualname = if class_prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{class_prefix}.{name}")
                };
                let id = if mod_name.is_empty() {
                    qualname.clone()
                } else {
                    format!("{mod_name}.{qualname}")
                };
                let start_line = outer.start_position().row + 1;
                let def_line = def_node.start_position().row + 1;
                let end_line = outer.end_position().row + 1;
                let body_start_line = first_body_line_after_docstring(def_node, src, def_line);
                let sig = Some(parse_signature(def_node, src, !class_prefix.is_empty()));
                let mut calls = Vec::new();
                collect_calls(def_node, src, &mut calls);
                let snippet = extract_body_aware_snippet(
                    outer,
                    src,
                    src_lines,
                    start_line,
                    body_start_line,
                    end_line,
                    3800,
                );
                let mut tokens =
                    tokenize_ident_and_subwords(&format!("{} {}", id.replace('.', " "), snippet));
                collect_ast_shape_and_tokens(def_node, src, &mut tokens);
                out.push((
                    DefNode {
                        id,
                        name,
                        qualname,
                        file: rel_path.to_string(),
                        kind: if is_async_fn {
                            "async_function".to_string()
                        } else {
                            "function".to_string()
                        },
                        start_line,
                        def_line,
                        body_start_line,
                        end_line,
                        sig,
                        calls,
                        bases: Vec::new(),
                        tokens,
                    },
                    is_overload,
                ));
                // Also index nested functions/classes if any
                if let Some(body) = def_node.child_by_field_name("body") {
                    walk_defs(body, src, src_lines, rel_path, mod_name, class_prefix, out);
                }
            }
            "class_definition" => {
                let Some(name_node) = def_node.child_by_field_name("name") else {
                    continue;
                };
                let name = node_text(name_node, src).to_string();
                let qualname = if class_prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{class_prefix}.{name}")
                };
                let id = if mod_name.is_empty() {
                    qualname.clone()
                } else {
                    format!("{mod_name}.{qualname}")
                };
                let start_line = outer.start_position().row + 1;
                let def_line = def_node.start_position().row + 1;
                let end_line = outer.end_position().row + 1;
                let body_start_line = first_body_line_after_docstring(def_node, src, def_line);
                let mut bases = Vec::new();
                if let Some(sup) = def_node.child_by_field_name("superclasses") {
                    let mut sc = sup.walk();
                    for b in sup.named_children(&mut sc) {
                        let btxt = node_text(b, src).trim();
                        if let Some(last) = btxt.split('[').next() {
                            let base_ident = last.rsplit('.').next().unwrap_or(last).trim();
                            if !base_ident.is_empty() && base_ident != "object" {
                                bases.push(base_ident.to_string());
                            }
                        }
                    }
                }
                let mut calls = Vec::new();
                collect_calls(def_node, src, &mut calls);
                let snippet = extract_body_aware_snippet(
                    outer,
                    src,
                    src_lines,
                    start_line,
                    body_start_line,
                    end_line,
                    2200,
                );
                let mut tokens =
                    tokenize_ident_and_subwords(&format!("{} {}", id.replace('.', " "), snippet));
                collect_ast_shape_and_tokens(def_node, src, &mut tokens);
                out.push((
                    DefNode {
                        id,
                        name,
                        qualname: qualname.clone(),
                        file: rel_path.to_string(),
                        kind: "class".to_string(),
                        start_line,
                        def_line,
                        body_start_line,
                        end_line,
                        sig: None,
                        calls,
                        bases,
                        tokens,
                    },
                    false,
                ));
                if let Some(body) = def_node.child_by_field_name("body") {
                    walk_defs(body, src, src_lines, rel_path, mod_name, &qualname, out);
                }
            }
            _ => {
                if child.kind() == "if_statement"
                    && let Some(cond) = child.child_by_field_name("condition")
                {
                    let cond_txt = node_text(cond, src);
                    if cond_txt.contains("__name__") && cond_txt.contains("__main__") {
                        continue;
                    }
                }
                walk_defs(child, src, src_lines, rel_path, mod_name, class_prefix, out);
            }
        }
    }
}

/// Build the full `RepoIndex` (AST definitions, Call & Inheritance Graph, `tests_of` links, and
/// mean-centred 256-d structural embeddings) in parallel across all `.py` files.
pub fn build_repo_index(ws: &Path) -> Result<RepoIndex> {
    let mut py_files = Vec::new();
    for entry in walkdir::WalkDir::new(ws)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            if e.depth() > 0 && name.starts_with('.') {
                return false;
            }
            !matches!(
                name.as_ref(),
                ".git"
                    | ".venv"
                    | "venv"
                    | "__pycache__"
                    | ".mypy_cache"
                    | ".pytest_cache"
                    | ".tox"
                    | "node_modules"
                    | "build"
                    | "dist"
            )
        })
        .flatten()
    {
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|s| s.to_str()) == Some("py")
            && let Ok(rel) = entry.path().strip_prefix(ws)
        {
            let rel_str = crate::clean_rel_arg(&rel.to_string_lossy());
            py_files.push((rel_str, entry.path().to_path_buf()));
        }
    }
    py_files.sort_by(|a, b| a.0.cmp(&b.0));

    let parsed: Vec<(String, Vec<DefNode>, HashMap<String, String>)> = py_files
        .par_iter()
        .filter(|(rel, _)| !is_docs_path(rel))
        .filter_map(|(rel, full)| {
            let content = crate::read_lf(full).ok()?;
            let (defs, imps) = parse_python_file(rel, &content);
            Some((rel.clone(), defs, imps))
        })
        .collect();

    let mut index = RepoIndex {
        files: py_files.into_iter().map(|(r, _)| r).collect(),
        ..Default::default()
    };

    let mut file_imports: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut by_bare_name: HashMap<String, Vec<String>> = HashMap::new();
    let mut by_qualname: HashMap<String, Vec<String>> = HashMap::new();

    for (rel, defs, imps) in parsed {
        file_imports.insert(rel.clone(), imps);
        let mut ids = Vec::with_capacity(defs.len());
        for mut d in defs {
            if let Some(prev) = index.nodes.remove(&d.id) {
                let prev_span = prev.end_line.saturating_sub(prev.start_line);
                let cur_span = d.end_line.saturating_sub(d.start_line);
                if prev_span > cur_span {
                    let mut keep = prev;
                    keep.tokens.extend(d.tokens);
                    for c in d.calls {
                        if !keep.calls.contains(&c) {
                            keep.calls.push(c);
                        }
                    }
                    index.nodes.insert(keep.id.clone(), keep);
                    continue;
                }
                d.tokens.extend(prev.tokens);
                for c in prev.calls {
                    if !d.calls.contains(&c) {
                        d.calls.push(c);
                    }
                }
            } else {
                ids.push(d.id.clone());
                by_bare_name
                    .entry(d.name.clone())
                    .or_default()
                    .push(d.id.clone());
                by_qualname
                    .entry(d.qualname.clone())
                    .or_default()
                    .push(d.id.clone());
            }
            index.nodes.insert(d.id.clone(), d);
        }
        index.file_defs.insert(rel, ids);
    }

    // Build Call, Class<->Method, Base<->Subclass, and Test->Target edges
    let mut adj_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut callers_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut callees_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut sub_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut super_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut tests_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    let all_nodes: Vec<DefNode> = index.nodes.values().cloned().collect();
    for d in &all_nodes {
        // Class <-> Member edge
        if let Some((cls_qual, _)) = d.qualname.rsplit_once('.') {
            let mod_name = module_name(&d.file);
            let cls_id = if mod_name.is_empty() {
                cls_qual.to_string()
            } else {
                format!("{mod_name}.{cls_qual}")
            };
            if index.nodes.contains_key(&cls_id) {
                adj_set
                    .entry(d.id.clone())
                    .or_default()
                    .insert(cls_id.clone());
                adj_set.entry(cls_id).or_default().insert(d.id.clone());
            }
        }
        // Base <-> Subclass edge
        for base in &d.bases {
            if let Some(cands) = by_bare_name.get(base) {
                for base_id in cands {
                    if base_id != &d.id
                        && index.nodes.get(base_id).is_some_and(|n| n.kind == "class")
                    {
                        sub_set
                            .entry(base_id.clone())
                            .or_default()
                            .insert(d.id.clone());
                        super_set
                            .entry(d.id.clone())
                            .or_default()
                            .insert(base_id.clone());
                        adj_set
                            .entry(base_id.clone())
                            .or_default()
                            .insert(d.id.clone());
                        adj_set
                            .entry(d.id.clone())
                            .or_default()
                            .insert(base_id.clone());
                    }
                }
            }
        }
        // Call edges
        let imps = file_imports.get(&d.file);
        let caller_is_test = is_test_path(&d.file);
        for (recv, cname) in &d.calls {
            let mut targets = Vec::new();
            if let Some(r) = recv {
                if r == "self" {
                    if let Some((cls_qual, _)) = d.qualname.rsplit_once('.') {
                        let q = format!("{cls_qual}.{cname}");
                        if let Some(cands) = by_qualname.get(&q) {
                            targets.extend(cands.iter().cloned());
                        }
                    }
                } else if let Some(imp_map) = imps
                    && let Some(mapped) = imp_map.get(r)
                {
                    let full_q = format!("{mapped}.{cname}");
                    if index.nodes.contains_key(&full_q) {
                        targets.push(full_q);
                    }
                }
            }
            if targets.is_empty()
                && let Some(imp_map) = imps
                && let Some(mapped) = imp_map.get(cname)
                && index.nodes.contains_key(mapped)
            {
                targets.push(mapped.clone());
            }
            if targets.is_empty()
                && let Some(cands) = by_bare_name.get(cname)
            {
                // Prefer same-file target, or unique repo-wide target
                let same_file: Vec<String> = cands
                    .iter()
                    .filter(|id| index.nodes.get(*id).is_some_and(|n| n.file == d.file))
                    .cloned()
                    .collect();
                if !same_file.is_empty() {
                    targets.extend(same_file);
                } else if cands.len() <= 3 {
                    targets.extend(cands.iter().cloned());
                }
            }
            for t in targets {
                if t == d.id {
                    continue;
                }
                if caller_is_test {
                    tests_set
                        .entry(t.clone())
                        .or_default()
                        .insert(d.file.clone());
                } else if !is_test_path(index.nodes.get(&t).map(|n| n.file.as_str()).unwrap_or(""))
                {
                    callers_set
                        .entry(t.clone())
                        .or_default()
                        .insert(d.id.clone());
                    callees_set
                        .entry(d.id.clone())
                        .or_default()
                        .insert(t.clone());
                    adj_set.entry(d.id.clone()).or_default().insert(t.clone());
                    adj_set.entry(t).or_default().insert(d.id.clone());
                }
            }
        }
    }

    // 2-hop subclass closure and superclass closure
    let base_keys: Vec<String> = sub_set.keys().cloned().collect();
    for b in base_keys {
        let hop1: Vec<String> = sub_set.get(&b).into_iter().flatten().cloned().collect();
        for s1 in hop1 {
            if let Some(hop2) = sub_set.get(&s1).cloned() {
                for s2 in &hop2 {
                    super_set.entry(s2.clone()).or_default().insert(b.clone());
                }
                sub_set.entry(b.clone()).or_default().extend(hop2);
            }
        }
    }

    // Method override propagation: if `Sub` subclasses `Base`, link `Sub.method` <-> `Base.method`
    for (sub_id, bases) in &super_set {
        let sub_prefix = format!("{sub_id}.");
        let sub_methods: Vec<(String, String)> = all_nodes
            .iter()
            .filter(|n| {
                n.id.starts_with(&sub_prefix)
                    && (n.kind == "function" || n.kind == "async_function")
            })
            .map(|n| (n.name.clone(), n.id.clone()))
            .collect();
        for base_id in bases {
            let base_prefix = format!("{base_id}.");
            let base_methods: HashMap<String, String> = all_nodes
                .iter()
                .filter(|n| {
                    n.id.starts_with(&base_prefix)
                        && (n.kind == "function" || n.kind == "async_function")
                })
                .map(|n| (n.name.clone(), n.id.clone()))
                .collect();
            for (mname, sm_id) in &sub_methods {
                if let Some(bm_id) = base_methods.get(mname) {
                    adj_set
                        .entry(sm_id.clone())
                        .or_default()
                        .insert(bm_id.clone());
                    adj_set
                        .entry(bm_id.clone())
                        .or_default()
                        .insert(sm_id.clone());
                    callers_set
                        .entry(bm_id.clone())
                        .or_default()
                        .insert(sm_id.clone());
                    callees_set
                        .entry(sm_id.clone())
                        .or_default()
                        .insert(bm_id.clone());
                }
            }
        }
    }

    for (k, v) in adj_set {
        index.adj.insert(k, v.into_iter().collect());
    }
    for (k, v) in callers_set {
        index.callers.insert(k, v.into_iter().collect());
    }
    for (k, v) in callees_set {
        index.callees.insert(k, v.into_iter().collect());
    }
    for (k, v) in sub_set {
        index.subclasses.insert(k, v.into_iter().collect());
    }
    for (k, v) in super_set {
        index.superclasses.insert(k, v.into_iter().collect());
    }
    for (k, v) in tests_set {
        index.tests_of.insert(k, v.into_iter().collect());
    }

    // Compute mean-centred 2-hop structural embeddings for non-test definitions
    index.embeddings = compute_whitened_embeddings(&index);
    Ok(index)
}

/// Compute 2-hop structural-token embeddings for every non-test definition, subtract the
/// repository mean vector, and L2-normalize (`\hat{v}_i = (v_i - \bar{v}) / ||v_i - \bar{v}||_2`).
/// Removing the mean keeps cosine similarity discriminative: raw vectors share a large common component.
pub fn compute_whitened_embeddings(index: &RepoIndex) -> BTreeMap<String, Vec<f32>> {
    let non_test_ids: Vec<String> = index
        .nodes
        .iter()
        .filter(|(_, n)| !is_test_path(&n.file))
        .map(|(id, _)| id.clone())
        .collect();
    if non_test_ids.is_empty() {
        return BTreeMap::new();
    }
    let pos: HashMap<&str, usize> = non_test_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    let mut x: Vec<Vec<f32>> = non_test_ids
        .iter()
        .map(|id| {
            let mut v = vec![0.0f32; EMBED_DIM];
            if let Some(node) = index.nodes.get(id) {
                for tok in &node.tokens {
                    let idx = (fnv1a(tok.as_bytes()) as usize) % EMBED_DIM;
                    v[idx] += 1.0;
                }
            }
            for e in &mut v {
                *e = e.ln_1p();
            }
            v
        })
        .collect();

    let nbrs: Vec<Vec<usize>> = non_test_ids
        .iter()
        .map(|id| {
            index
                .adj
                .get(id)
                .into_iter()
                .flatten()
                .filter_map(|nb| pos.get(nb.as_str()).copied())
                .collect()
        })
        .collect();

    // 2-hop structural propagation: v_i <- (v_i + 0.5 * mean(N_1)) then second hop
    for weight in [0.5f32, 0.25f32] {
        let next: Vec<Vec<f32>> = (0..x.len())
            .map(|i| {
                if nbrs[i].is_empty() {
                    return x[i].clone();
                }
                let mut v = x[i].clone();
                let inv_deg = weight / (nbrs[i].len() as f32);
                for &j in &nbrs[i] {
                    for (a, &b) in v.iter_mut().zip(&x[j]) {
                        *a += inv_deg * b;
                    }
                }
                v
            })
            .collect();
        x = next;
    }

    // Subtract repository mean vector with small-sample shrinkage (alpha = (n-1)/(n+15)) and L2-normalize.
    // With many nodes alpha -> 1.0 (full centring); in a module of a few nodes the shrinkage
    // keeps a small cluster of similar functions from defining the mean.
    let n = x.len() as f32;
    let alpha = (n - 1.0).max(0.0) / (n + 15.0);
    let mut mean = vec![0.0f32; EMBED_DIM];
    for row in &x {
        for (m, &val) in mean.iter_mut().zip(row) {
            *m += val / n;
        }
    }

    let mut out = BTreeMap::new();
    for (id, mut row) in non_test_ids.into_iter().zip(x) {
        let mut norm_sq = 0.0f32;
        for (val, &m) in row.iter_mut().zip(&mean) {
            *val -= alpha * m;
            norm_sq += (*val) * (*val);
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for val in &mut row {
            *val /= norm;
        }
        out.insert(id, row);
    }
    out
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Find the closest structural sibling of `seed_id` using the mean-centred structural embeddings.
/// Excludes direct callers/callees (`adj`) since those are already surfaced via the Call Graph,
/// allowing `find_structural_sibling` to uncover parallel/duplicate implementations (e.g. sync vs async).
pub fn find_structural_sibling(index: &RepoIndex, seed_id: &str) -> Option<(String, f32)> {
    let seed_vec = index.embeddings.get(seed_id)?;
    let seed_node = index.nodes.get(seed_id)?;
    let direct_adj: HashSet<&str> = index
        .adj
        .get(seed_id)
        .into_iter()
        .flatten()
        .map(String::as_str)
        .collect();
    let mut best: Option<(String, f32)> = None;
    for (id, vec) in &index.embeddings {
        if id == seed_id || direct_adj.contains(id.as_str()) {
            continue;
        }
        let Some(cand) = index.nodes.get(id) else {
            continue;
        };
        // Prefer same kind (function vs class)
        if (cand.kind == "class") != (seed_node.kind == "class") {
            continue;
        }
        let mut sim = cosine_sim(seed_vec, vec);
        if cand.file == seed_node.file {
            sim += 0.08; // Same-module sibling bonus
        }
        if best.as_ref().is_none_or(|(_, b)| sim > *b) {
            best = Some((id.clone(), sim));
        }
    }
    best.filter(|(_, sim)| *sim >= 0.25)
}

/// Check Python source for a syntax or indentation error with CPython `ast.parse`,
/// falling back to `tree-sitter-python` only if no Python interpreter can be spawned.
/// Returns `Some((1_indexed_line, snippet))` if a syntax error exists.
pub fn check_python_syntax(src_str: &str) -> Option<(usize, String)> {
    if let Some(res) = check_python_syntax_cpython(src_str) {
        return res;
    }
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("tree-sitter-python");
    let src = src_str.as_bytes();
    let tree = parser.parse(src, None)?;
    let root = tree.root_node();
    if root.has_error() {
        return find_first_error_node(root, src);
    }
    if src.contains(&b'\\') {
        return find_illegal_backslash_outside_string(root, src_str);
    }
    None
}

fn check_python_syntax_cpython(src_str: &str) -> Option<Option<(usize, String)>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    const PY_SYNTAX_SNIPPET: &str = "import ast, sys\nsrc = sys.stdin.read()\ntry:\n    ast.parse(src)\nexcept SyntaxError as e:\n    ln = e.lineno or 1\n    txt = (e.text or '').strip()[:80]\n    print(f'{ln}\\t{type(e).__name__}: {e.msg}' + (f' (`{txt}`)' if txt else ''))\n    sys.exit(2)\n";

    let bins: &[&str] = if cfg!(windows) {
        &["python", "python3"]
    } else {
        &["python3", "python"]
    };

    for &bin in bins {
        let Ok(mut child) = Command::new(bin)
            .args(["-I", "-c", PY_SYNTAX_SNIPPET])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(src_str.as_bytes());
        }
        let Ok(output) = child.wait_with_output() else {
            continue;
        };
        match output.status.code() {
            Some(0) => return Some(None),
            Some(2) => {
                let raw = String::from_utf8_lossy(&output.stdout);
                let first = raw.lines().next().unwrap_or("").trim();
                if let Some((ln_s, msg)) = first.split_once('\t') {
                    let ln = ln_s.parse::<usize>().unwrap_or(1).max(1);
                    return Some(Some((ln, msg.to_string())));
                }
                return Some(Some((1, first.to_string())));
            }
            _ => continue,
        }
    }
    None
}

fn find_illegal_backslash_outside_string(root: Node, src_str: &str) -> Option<(usize, String)> {
    let mut spans = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if matches!(n.kind(), "string" | "comment") {
            spans.push((n.start_byte(), n.end_byte()));
            continue;
        }
        let mut cur = n.walk();
        for c in n.children(&mut cur) {
            stack.push(c);
        }
    }
    let bytes = src_str.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        if b != b'\\' {
            continue;
        }
        if spans.iter().any(|&(s, e)| idx >= s && idx < e) {
            continue;
        }
        // Outside string/comment: backslash must be followed immediately by \n, \r, or EOF
        let next = bytes.get(idx + 1).copied();
        if !matches!(next, None | Some(b'\n' | b'\r')) {
            let line = bytes[..idx].iter().filter(|&&c| c == b'\n').count() + 1;
            let snippet: String = src_str
                .lines()
                .nth(line.saturating_sub(1))
                .unwrap_or("\\")
                .trim()
                .chars()
                .take(80)
                .collect();
            return Some((
                line,
                format!("unexpected character after backslash continuation: {snippet}"),
            ));
        }
    }
    None
}

fn find_first_error_node(n: Node, src: &[u8]) -> Option<(usize, String)> {
    if n.is_error() || n.is_missing() {
        let line = n.start_position().row + 1;
        let snippet: String = node_text(n, src)
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(80)
            .collect();
        return Some((line, snippet));
    }
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        if c.has_error()
            && let Some(err) = find_first_error_node(c, src)
        {
            return Some(err);
        }
    }
    Some((n.start_position().row + 1, "syntax error".to_string()))
}

/// Refuse a line-range edit that lies strictly inside a multi-line string literal (docstring / `Doc("""...""")`).
/// Returns `Some((str_start_line, str_end_line, enclosing_fn_body_start))` when `[start_line, end_line]` is inside a string.
pub fn inside_multiline_string(
    src_str: &str,
    start_line: usize,
    end_line: usize,
) -> Option<(usize, usize, Option<usize>)> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("tree-sitter-python");
    let src = src_str.as_bytes();
    let tree = parser.parse(src, None)?;
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        if n.kind() == "string" {
            let s = n.start_position().row + 1;
            let e = n.end_position().row + 1;
            if e > s && s < start_line && end_line < e {
                // Find enclosing function body start line if any
                let mut p = n.parent();
                let mut body_start = None;
                while let Some(curr) = p {
                    if curr.kind() == "function_definition" {
                        body_start = Some(first_body_line_after_docstring(
                            curr,
                            src,
                            curr.start_position().row + 1,
                        ));
                        break;
                    }
                    p = curr.parent();
                }
                return Some((s, e, body_start));
            }
        }
        let mut cur = n.walk();
        for c in n.children(&mut cur) {
            stack.push(c);
        }
    }
    None
}

/// Detect newly introduced undefined names or missing `self.<method>` definitions in `new_src` compared to `old_src`.
pub fn check_new_undefined_and_self_methods(
    rel_file: &str,
    old_src: &str,
    new_src: &str,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let old_undef = undefined_identifiers(old_src);
    let new_undef = undefined_identifiers(new_src);
    let added: Vec<String> = new_undef.difference(&old_undef).cloned().collect();
    if !added.is_empty() {
        let mut regular = Vec::new();
        for nm in added.into_iter().take(5) {
            let err_cand = format!("{nm}Error");
            if PYTHON_BUILTINS.contains(&err_cand.as_str()) {
                warnings.push(format!(
                    "{rel_file}: `{nm}` is not defined — Python's built-in exception is `{err_cand}` (use `{err_cand}` directly; do NOT import `{nm}`!)"
                ));
            } else if nm == "KeyboardException" {
                warnings.push(format!(
                    "{rel_file}: `KeyboardException` is not defined — Python's built-in is `KeyboardInterrupt` (no import needed!)"
                ));
            } else {
                regular.push(nm);
            }
        }
        if !regular.is_empty() {
            warnings.push(format!(
                "{rel_file}: {} used but never defined or imported",
                regular.join(", ")
            ));
        }
    }
    let old_self = missing_self_calls(old_src);
    let new_self = missing_self_calls(new_src);
    let added_self: Vec<String> = new_self.difference(&old_self).cloned().collect();
    if !added_self.is_empty() {
        warnings.push(format!(
            "{rel_file}: {} called on self but never defined in the class",
            added_self
                .into_iter()
                .take(5)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    warnings
}

pub(crate) const PYTHON_BUILTINS: &[&str] = &[
    "abs",
    "aiter",
    "all",
    "anext",
    "any",
    "ascii",
    "bin",
    "bool",
    "breakpoint",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "__import__",
    "__name__",
    "__file__",
    "__doc__",
    "__package__",
    "__spec__",
    "__path__",
    "__builtins__",
    "__all__",
    "__dict__",
    "__class__",
    "True",
    "False",
    "None",
    "Ellipsis",
    "NotImplemented",
    "Exception",
    "BaseException",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "StopIteration",
    "StopAsyncIteration",
    "GeneratorExit",
    "AssertionError",
    "ImportError",
    "ModuleNotFoundError",
    "OSError",
    "IOError",
    "FileNotFoundError",
    "FileExistsError",
    "PermissionError",
    "TimeoutError",
    "LookupError",
    "ArithmeticError",
    "ZeroDivisionError",
    "OverflowError",
    "FloatingPointError",
    "NameError",
    "UnboundLocalError",
    "SyntaxError",
    "IndentationError",
    "SystemError",
    "SystemExit",
    "KeyboardInterrupt",
    "MemoryError",
    "RecursionError",
    "NotImplementedError",
    "Warning",
    "UserWarning",
    "DeprecationWarning",
    "FutureWarning",
    "RuntimeWarning",
    "SyntaxWarning",
    "EncodingWarning",
    "BytesWarning",
    "ResourceWarning",
    "ConnectionError",
    "BrokenPipeError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "BlockingIOError",
    "ChildProcessError",
    "InterruptedError",
    "IsADirectoryError",
    "NotADirectoryError",
    "ProcessLookupError",
    "UnicodeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeTranslateError",
    "BufferError",
    "EOFError",
    "ReferenceError",
    "ExceptionGroup",
    "BaseExceptionGroup",
    "TYPE_CHECKING",
    "reveal_type",
    "self",
    "cls",
];

fn undefined_identifiers(src_str: &str) -> BTreeSet<String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("tree-sitter-python");
    let src = src_str.as_bytes();
    let Some(tree) = parser.parse(src, None) else {
        return BTreeSet::new();
    };
    let root = tree.root_node();
    // Star imports (`from x import *`) bind arbitrary names
    if src_str.contains("import *") {
        return BTreeSet::new();
    }
    let deferred_annotations = src_str.contains("from __future__ import annotations");
    let mut bound: HashSet<String> = PYTHON_BUILTINS.iter().map(|s| (*s).to_string()).collect();
    let mut used: BTreeSet<String> = BTreeSet::new();

    collect_bound_and_used(
        root,
        src,
        &mut bound,
        &mut used,
        false,
        deferred_annotations,
    );
    used.into_iter().filter(|u| !bound.contains(u)).collect()
}

fn collect_bound_and_used(
    n: Node,
    src: &[u8],
    bound: &mut HashSet<String>,
    used: &mut BTreeSet<String>,
    in_lhs: bool,
    deferred_ann: bool,
) {
    match n.kind() {
        "function_definition" | "class_definition" => {
            if let Some(nm) = n.child_by_field_name("name") {
                bound.insert(node_text(nm, src).to_string());
            }
            if let Some(params) = n.child_by_field_name("parameters") {
                collect_bound_and_used(params, src, bound, used, true, deferred_ann);
            }
            if let Some(tp) = n.child_by_field_name("type_parameters") {
                collect_bound_and_used(tp, src, bound, used, true, deferred_ann);
            }
            let mut cur = n.walk();
            for c in n.children(&mut cur) {
                if deferred_ann && n.child_by_field_name("return_type") == Some(c) {
                    continue;
                }
                if n.child_by_field_name("parameters") == Some(c)
                    || n.child_by_field_name("type_parameters") == Some(c)
                {
                    continue;
                }
                collect_bound_and_used(c, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "typed_parameter" | "typed_default_parameter" => {
            if let Some(name_node) = n.child_by_field_name("name").or_else(|| n.named_child(0)) {
                collect_bound_and_used(name_node, src, bound, used, true, deferred_ann);
            }
            if !deferred_ann && let Some(tp) = n.child_by_field_name("type") {
                collect_bound_and_used(tp, src, bound, used, false, deferred_ann);
            }
            if let Some(val) = n.child_by_field_name("value") {
                collect_bound_and_used(val, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "default_parameter" => {
            if let Some(name_node) = n.child_by_field_name("name") {
                collect_bound_and_used(name_node, src, bound, used, true, deferred_ann);
            }
            if let Some(val) = n.child_by_field_name("value") {
                collect_bound_and_used(val, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "lambda" => {
            if let Some(params) = n.child_by_field_name("parameters") {
                collect_bound_and_used(params, src, bound, used, true, deferred_ann);
            }
            if let Some(body) = n.child_by_field_name("body") {
                collect_bound_and_used(body, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "list_comprehension"
        | "dictionary_comprehension"
        | "set_comprehension"
        | "generator_expression" => {
            let mut cur = n.walk();
            let children: Vec<Node> = n.children(&mut cur).collect();
            for c in &children {
                if c.kind() == "for_in_clause" {
                    collect_bound_and_used(*c, src, bound, used, false, deferred_ann);
                }
            }
            for c in &children {
                if c.kind() != "for_in_clause" {
                    collect_bound_and_used(*c, src, bound, used, false, deferred_ann);
                }
            }
            return;
        }
        "import_statement" => {
            let mut cur = n.walk();
            for c in n.named_children(&mut cur) {
                if c.kind() == "dotted_name" {
                    let top = node_text(c, src).split('.').next().unwrap_or("");
                    bound.insert(top.to_string());
                } else if c.kind() == "aliased_import"
                    && let Some(a) = c.child_by_field_name("alias")
                {
                    bound.insert(node_text(a, src).to_string());
                }
            }
            return;
        }
        "import_from_statement" => {
            let mut cur = n.walk();
            for c in n.children_by_field_name("name", &mut cur) {
                if c.kind() == "dotted_name" {
                    let top = node_text(c, src).split('.').next().unwrap_or("");
                    bound.insert(top.to_string());
                } else if c.kind() == "aliased_import"
                    && let Some(a) = c.child_by_field_name("alias")
                {
                    bound.insert(node_text(a, src).to_string());
                }
            }
            return;
        }
        "assignment" | "augmented_assignment" => {
            if let Some(left) = n.child_by_field_name("left") {
                collect_bound_and_used(left, src, bound, used, true, deferred_ann);
            }
            if let Some(right) = n.child_by_field_name("right") {
                collect_bound_and_used(right, src, bound, used, false, deferred_ann);
            }
            if !deferred_ann && let Some(tp) = n.child_by_field_name("type") {
                collect_bound_and_used(tp, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "named_expression" => {
            if let Some(name) = n.child_by_field_name("name") {
                collect_bound_and_used(name, src, bound, used, true, deferred_ann);
            }
            if let Some(val) = n.child_by_field_name("value") {
                collect_bound_and_used(val, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "for_statement" | "for_in_clause" => {
            if let Some(left) = n.child_by_field_name("left") {
                collect_bound_and_used(left, src, bound, used, true, deferred_ann);
            }
        }
        "as_pattern" => {
            if let Some(alias) = n.child_by_field_name("alias") {
                collect_bound_and_used(alias, src, bound, used, true, deferred_ann);
            }
        }
        "case_pattern" => {
            collect_bound_and_used_children(n, src, bound, used, true, deferred_ann);
            return;
        }
        "except_clause" => {
            let mut cur = n.walk();
            let children: Vec<Node> = n.named_children(&mut cur).collect();
            if children.len() >= 2 {
                collect_bound_and_used(children[1], src, bound, used, true, deferred_ann);
            }
        }
        "global_statement" | "nonlocal_statement" => {
            let mut cur = n.walk();
            for c in n.named_children(&mut cur) {
                if c.kind() == "identifier" {
                    bound.insert(node_text(c, src).to_string());
                }
            }
            return;
        }
        "keyword_argument" => {
            if let Some(val) = n.child_by_field_name("value") {
                collect_bound_and_used(val, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "attribute" => {
            if let Some(obj) = n.child_by_field_name("object") {
                collect_bound_and_used(obj, src, bound, used, false, deferred_ann);
            }
            return;
        }
        "identifier" => {
            let id = node_text(n, src);
            if in_lhs {
                bound.insert(id.to_string());
            } else {
                used.insert(id.to_string());
            }
            return;
        }
        _ => {}
    }
    collect_bound_and_used_children(n, src, bound, used, in_lhs, deferred_ann);
}

fn collect_bound_and_used_children(
    n: Node,
    src: &[u8],
    bound: &mut HashSet<String>,
    used: &mut BTreeSet<String>,
    in_lhs: bool,
    deferred_ann: bool,
) {
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        collect_bound_and_used(c, src, bound, used, in_lhs, deferred_ann);
    }
}

fn missing_self_calls(src_str: &str) -> BTreeSet<String> {
    let (defs, _) = parse_python_file("mod.py", src_str);
    let mut class_bases: HashMap<String, Vec<String>> = HashMap::new();
    let mut class_members: HashMap<String, HashSet<String>> = HashMap::new();
    let mut class_calls: HashMap<String, Vec<String>> = HashMap::new();

    for d in &defs {
        if d.kind == "class" {
            class_bases.insert(d.qualname.clone(), d.bases.clone());
            class_members.entry(d.qualname.clone()).or_default();
        } else if let Some((cls, meth)) = d.qualname.rsplit_once('.') {
            class_members
                .entry(cls.to_string())
                .or_default()
                .insert(meth.to_string());
            for (recv, cname) in &d.calls {
                if recv.as_deref() == Some("self") {
                    class_calls
                        .entry(cls.to_string())
                        .or_default()
                        .push(cname.clone());
                }
            }
        }
    }
    // Also scan `self.attr = ...` assignments and class-level names (dataclass fields, class
    // attributes) inside class bodies, so `self.<module function>(...)` in a dataclass is caught.
    let self_assign = Regex::new(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*[:=]").expect("regex");
    let class_level = Regex::new(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::|=[^=])").expect("regex");
    let lines: Vec<&str> = src_str.lines().collect();
    for d in &defs {
        if d.kind == "class" {
            let slice = &lines[d.def_line.saturating_sub(1)..d.end_line.min(lines.len())];
            let body = slice.join("\n");
            let members = class_members.entry(d.qualname.clone()).or_default();
            for cap in self_assign.captures_iter(&body) {
                members.insert(cap[1].to_string());
            }
            let body_indent = slice
                .iter()
                .skip(1)
                .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .map(|l| l.len() - l.trim_start().len());
            if let Some(ind) = body_indent {
                for l in slice.iter().skip(1) {
                    if l.len() - l.trim_start().len() == ind
                        && let Some(cap) = class_level.captures(l.trim_start())
                    {
                        members.insert(cap[1].to_string());
                    }
                }
            }
        }
    }

    let mut out = BTreeSet::new();
    for (cls, calls) in class_calls {
        let Some(bases) = class_bases.get(&cls) else {
            continue;
        };
        if bases.iter().any(|b| !class_bases.contains_key(b)) {
            continue; // Inherits from an external class that may define the method
        }
        let mut have = class_members.get(&cls).cloned().unwrap_or_default();
        for b in bases {
            if let Some(bm) = class_members.get(b) {
                have.extend(bm.iter().cloned());
            }
        }
        if have.contains("__getattr__") || have.contains("__getattribute__") {
            continue; // dynamic attributes
        }
        for c in calls {
            if !c.starts_with("__") && !have.contains(&c) {
                out.insert(format!("{cls}.{c}"));
            }
        }
    }
    out
}

/// Detect functions/methods in `rel_file` whose parameter signature sets changed between `old_src` and `new_src`.
/// Groups signatures by `qualname` so `@overload` blocks do not trigger false-positive changes.
pub fn changed_signatures(
    rel_file: &str,
    old_src: &str,
    new_src: &str,
) -> Vec<(String, String, Signature, Signature)> {
    let (old_defs, _) = parse_python_file(rel_file, old_src);
    let (new_defs, _) = parse_python_file(rel_file, new_src);

    let mut old_by_qual: BTreeMap<String, Vec<Signature>> = BTreeMap::new();
    for d in old_defs {
        if let Some(sig) = d.sig {
            old_by_qual.entry(d.qualname).or_default().push(sig);
        }
    }

    let mut new_by_qual: BTreeMap<String, (String, Vec<Signature>)> = BTreeMap::new();
    for d in new_defs {
        if let Some(sig) = d.sig {
            let entry = new_by_qual
                .entry(d.qualname)
                .or_insert_with(|| (d.name, Vec::new()));
            entry.1.push(sig);
        }
    }

    let mut out = Vec::new();
    for (qual, (bare_name, new_sigs)) in new_by_qual {
        if let Some(old_sigs) = old_by_qual.get(&qual)
            && old_sigs != &new_sigs
            && let (Some(old_last), Some(new_last)) = (old_sigs.last(), new_sigs.last())
            && old_last != new_last
        {
            out.push((qual, bare_name, old_last.clone(), new_last.clone()));
        }
    }
    out
}

/// Detect an existing method that gained or lost `@property` / `@cached_property`: every
/// `obj.name` (or `obj.name()`) access in the repo breaks.
pub fn check_property_toggles(
    rel_file: &str,
    old_src: &str,
    new_src: &str,
    issue_text: &str,
) -> Vec<String> {
    fn props(src: &str) -> HashMap<String, bool> {
        let (defs, _) = parse_python_file("mod.py", src);
        let lines: Vec<&str> = src.lines().collect();
        defs.into_iter()
            .filter(|d| d.kind != "class" && d.qualname.contains('.'))
            .map(|d| {
                let deco = &lines
                    [d.start_line.saturating_sub(1)..d.def_line.saturating_sub(1).min(lines.len())];
                let is_prop = deco.iter().any(|l| {
                    let t = l.trim_start();
                    t.starts_with('@')
                        && (t.ends_with("property") || t.contains("property("))
                        && !t.contains(".setter")
                        && !t.contains(".deleter")
                });
                (d.qualname, is_prop)
            })
            .collect()
    }
    if old_src.trim().is_empty() {
        return Vec::new();
    }
    let issue_low = issue_text.to_ascii_lowercase();
    let old = props(old_src);
    let mut out = Vec::new();
    for (qual, now) in props(new_src) {
        let Some(&was) = old.get(&qual) else { continue };
        let name = qual.rsplit('.').next().unwrap_or(&qual);
        if was == now
            || (issue_low.contains("property") && issue_low.contains(&name.to_ascii_lowercase()))
        {
            continue;
        }
        out.push(if was {
            format!("{rel_file}: `{qual}` is no longer a @property, so every existing `obj.{name}` access now gets a bound method! Keep the decorator; in a repro read it as `obj.{name}` (no parentheses)")
        } else {
            format!("{rel_file}: `{qual}` became a @property, so every existing `obj.{name}()` call now fails! Keep it a method")
        });
    }
    out.sort();
    out
}

/// Detect accidental deletion of existing public functions/classes/methods (`!name.starts_with('_')`)
/// when `issue_text` does not explicitly request removing/deleting/renaming that symbol.
pub fn check_deleted_public_defs(
    rel_file: &str,
    old_src: &str,
    new_src: &str,
    issue_text: &str,
) -> Vec<String> {
    if old_src.trim().is_empty() {
        return Vec::new();
    }
    let (old_defs, _) = parse_python_file(rel_file, old_src);
    let (new_defs, _) = parse_python_file(rel_file, new_src);
    let new_quals: HashSet<&str> = new_defs.iter().map(|d| d.qualname.as_str()).collect();
    let new_classes: HashSet<&str> = new_defs
        .iter()
        .filter(|d| d.kind == "class")
        .map(|d| d.qualname.as_str())
        .collect();

    let issue_low = issue_text.to_ascii_lowercase();
    let removal_intent = ["remove", "delete", "drop", "rename", "deprecate"]
        .iter()
        .any(|kw| issue_low.contains(kw));
    // a rename written as "`A.old` -> `.new`" names `old` for removal without any keyword
    let renamed: Vec<String> = crate::debug::rename_pairs(issue_text)
        .into_iter()
        .map(|(old, _)| old)
        .collect();

    let mut deleted = Vec::new();
    for d in &old_defs {
        // dunders (`__init__`, `__call__`) are public API: overwriting `__init__` with another method
        // counts as a deletion
        let dunder = d.name.starts_with("__") && d.name.ends_with("__");
        if (d.name.starts_with('_') && !dunder) || new_quals.contains(d.qualname.as_str()) {
            continue;
        }
        // Only check top-level definitions or methods whose enclosing class still exists
        if let Some((parent_cls, _)) = d.qualname.rsplit_once('.')
            && !new_classes.contains(parent_cls)
        {
            continue;
        }
        if (removal_intent && issue_low.contains(&d.name.to_ascii_lowercase()))
            || renamed.contains(&d.name)
        {
            continue;
        }
        if !deleted.contains(&d.qualname) {
            deleted.push(d.qualname.clone());
        }
    }
    if deleted.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{rel_file}: existing public definition(s) {} were accidentally deleted/overwritten! Keep existing functions/classes intact and add your new code alongside them",
            deleted
                .into_iter()
                .take(4)
                .map(|q| format!("`{q}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )]
    }
}

/// Detect `def method(self, ...)` / `def method(cls, ...)` definitions that were accidentally
/// indented inside the body of a preceding class method (valid Python syntax as a dead nested
/// function, so `check_python_syntax` alone does not catch it).
pub fn find_misindented_nested_methods(src: &str) -> Vec<(usize, String, String)> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .is_err()
    {
        return Vec::new();
    }
    let Some(tree) = parser.parse(src, None) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    walk_nested_methods(tree.root_node(), src.as_bytes(), None, false, &mut out);
    out
}

fn walk_nested_methods<'a>(
    node: Node<'a>,
    src: &'a [u8],
    outer_fn: Option<&'a str>,
    in_class: bool,
    out: &mut Vec<(usize, String, String)>,
) {
    match node.kind() {
        "class_definition" => {
            if let Some(body) = node.child_by_field_name("body") {
                let mut cur = body.walk();
                for ch in body.children(&mut cur) {
                    walk_nested_methods(ch, src, None, true, out);
                }
            }
        }
        "function_definition" => {
            let name = node
                .child_by_field_name("name")
                .map(|n| node_text(n, src))
                .unwrap_or("");
            if in_class && let Some(outer) = outer_fn {
                let sig = parse_signature(node, src, true);
                let takes_self_or_cls = sig.pos.first().is_some_and(|p| p == "self" || p == "cls");
                if takes_self_or_cls && name != outer {
                    out.push((
                        node.start_position().row + 1,
                        name.to_string(),
                        outer.to_string(),
                    ));
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                let next_outer = outer_fn.or(Some(name));
                let mut cur = body.walk();
                for ch in body.children(&mut cur) {
                    walk_nested_methods(ch, src, next_outer, in_class, out);
                }
            }
        }
        "decorated_definition" => {
            let mut cur = node.walk();
            for ch in node.children(&mut cur) {
                if matches!(ch.kind(), "class_definition" | "function_definition") {
                    walk_nested_methods(ch, src, outer_fn, in_class, out);
                }
            }
        }
        _ => {
            let mut cur = node.walk();
            for ch in node.children(&mut cur) {
                walk_nested_methods(ch, src, outer_fn, in_class, out);
            }
        }
    }
}
