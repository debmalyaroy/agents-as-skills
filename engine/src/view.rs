//! `/tmp/coder view`: AST symbol (`def`, `async def`, `class`, `Class.method`),
//! line-range and pattern reader with a hard token cap.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::ast_graph::{DefNode, is_docs_path, is_test_path, parse_python_file};
use crate::{EnvPaths, clip_tokens, format_numbered_slice, load_or_build_index};

/// Normalize a `symbol` argument when a model passes a full Python signature such as
/// `"def state_command(connection_name: str | None = ...) -> None:"`, `"class RichOption(Option):"`,
/// or copies a `view` banner like `"pkg/utils.py:71-97 (function create_model_field, file has 164 lines)"`.
pub fn normalize_symbol_query(raw: &str) -> String {
    let mut s = raw.trim();
    for ln in s.lines() {
        let t = ln.trim();
        if t.is_empty() || t.starts_with('@') || t.starts_with('#') {
            continue;
        }
        s = t;
        break;
    }
    s = s.trim_matches('-').trim();
    // Extract `<qualname>` if the model copied a `view`/`debug` banner `(function Foo.bar, ...)`
    for tag in [
        "(async_function ",
        "(async def ",
        "(function ",
        "(method ",
        "(class ",
        "(def ",
    ] {
        if let Some(idx) = s.find(tag) {
            let after = &s[idx + tag.len()..];
            let extracted: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            if !extracted.is_empty() {
                return extracted;
            }
        }
    }
    for kw in ["async def ", "def ", "class "] {
        if let Some(rest) = s.strip_prefix(kw) {
            s = rest.trim_start();
            break;
        }
    }
    if let Some((head, _)) = s.split_once('(') {
        s = head.trim();
    }
    if let Some((head, _)) = s.split_once('[') {
        s = head.trim();
    }
    if let Some((head, _)) = s.split_once("->") {
        s = head.trim();
    }
    // Strip accidental `pkg/mod.py:71-97` prefix if no symbol remained after `:`
    if let Some((left, right)) = s.rsplit_once(':')
        && left.ends_with(".py")
        && right
            .chars()
            .all(|c| c.is_ascii_digit() || c == '-' || c == '.')
    {
        return String::new();
    }
    s = s.trim_end_matches(':').trim();
    s.trim_matches(|c: char| c == '`' || c == '\'' || c == '"')
        .trim()
        .to_string()
}

/// Format a `DefNode` slice while automatically skipping giant parameter/function/class docstrings
/// (`d.body_start_line > d.start_line + 25`, e.g. `Annotated[Doc(...)]` parameters or long class docstrings)
/// so the executable body is always visible and never cut off inside a docstring
pub fn format_def_body_aware(content: &str, d: &DefNode, max_body_lines: usize) -> String {
    let total_lines = content.lines().count().max(1);
    if d.body_start_line > d.start_line + 25 && d.end_line > d.start_line + max_body_lines {
        let head_end = (d.start_line + 10).min(d.body_start_line.saturating_sub(1));
        let body_s = d.body_start_line.saturating_sub(2).max(head_end + 1);
        let body_e = d.end_line.min(d.body_start_line + max_body_lines);
        let mut out = format!(
            "--- {}:{}-{} ({} {} header, file has {} lines) ---\n",
            d.file, d.start_line, head_end, d.kind, d.qualname, total_lines
        );
        out.push_str(&format_numbered_slice(content, d.start_line, head_end));
        out.push_str(&format!(
            "--- {}:{}-{} ({} {} executable body; docstring lines {}..{} omitted — match `old_string` inside {}..{}!) ---\n",
            d.file,
            body_s,
            body_e,
            d.kind,
            d.qualname,
            head_end + 1,
            body_s.saturating_sub(1),
            body_s,
            body_e
        ));
        out.push_str(&format_numbered_slice(content, body_s, body_e));
        out
    } else {
        let e = d.end_line.min(d.start_line + max_body_lines);
        let mut out = format!(
            "--- {}:{}-{} ({} {}, file has {} lines) ---\n",
            d.file, d.start_line, e, d.kind, d.qualname, total_lines
        );
        out.push_str(&format_numbered_slice(content, d.start_line, e));
        out
    }
}

/// The same `view` again: serve the symbol's next 60-line window instead of the same clipped
/// text. `None` when the call already names a range.
pub fn page_repeated_view(
    env: &EnvPaths,
    args: &BTreeMap<String, String>,
    repeat_count: usize,
) -> Result<Option<String>> {
    if ["start", "start_line", "end", "end_line", "targets"]
        .iter()
        .any(|k| args.contains_key(*k))
    {
        return Ok(None);
    }
    let first = run_view(env, args)?;
    let re = regex::Regex::new(r"(?m)^--- ([^:\n]+\.py):(\d+)-(\d+) \(").expect("regex");
    let Some(c) = re.captures(&first) else {
        return Ok(None);
    };
    let file = c[1].to_string();
    let a: usize = c[2].parse().unwrap_or(1);
    let b: usize = c[3].parse().unwrap_or(a);
    let windows = (b.saturating_sub(a) / 55) + 1;
    let k = (repeat_count - 1) % windows;
    let s = a + 55 * k;
    let e = (s + 59).min(b.max(s));
    let mut sub = BTreeMap::new();
    sub.insert("file".to_string(), file.clone());
    sub.insert("start".to_string(), s.to_string());
    sub.insert("end".to_string(), e.to_string());
    let body = run_view(env, &sub)?;
    Ok(Some(format!(
        "REPEATED VIEW: same arguments as before, so here is window {}/{windows} of `{file}` {a}-{b} (lines {s}-{e}). For other lines pass `\"start\"`/`\"end\"`; once you have the lines you need, edit with `scripts/patch.py`.\n{body}",
        k + 1
    )))
}

pub fn run_view(env: &EnvPaths, args: &BTreeMap<String, String>) -> Result<String> {
    if let Some(targets) = args.get("targets").filter(|s| !s.trim().is_empty()) {
        let mut out = String::new();
        for spec in targets.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            out.push_str(&view_one_target_spec(env, spec)?);
            out.push('\n');
        }
        return Ok(clip_tokens(&out, 1100));
    }

    let raw_file_opt = args
        .get("file")
        .or_else(|| args.get("path"))
        .or_else(|| args.get("filepath"))
        .or_else(|| args.get("file_path"))
        .map(|s| s.trim());
    if matches!(raw_file_opt, Some("*" | "." | "./")) {
        return list_workspace_files(env, "");
    }

    let mut file_clean = raw_file_opt.map(crate::clean_rel_arg).unwrap_or_default();
    let mut symbol_owned = args
        .get("symbol")
        .or_else(|| args.get("name"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut pattern_owned = args
        .get("pattern")
        .or_else(|| args.get("grep"))
        .or_else(|| args.get("search"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // Recover when the model passed `query="sym pkg/mod.py"` to `view`
    if file_clean.is_empty()
        && symbol_owned.is_empty()
        && pattern_owned.is_empty()
        && let Some(q) = args
            .get("query")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
    {
        let mut rem = Vec::new();
        for tok in q.split_whitespace() {
            if file_clean.is_empty() && crate::looks_like_file_arg(tok) {
                file_clean = crate::clean_rel_arg(tok);
            } else {
                rem.push(tok);
            }
        }
        if !rem.is_empty() {
            let all_ident_like = rem.iter().all(|tok| {
                tok.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ','))
            });
            if file_clean.is_empty() && rem.len() == 1 {
                symbol_owned = rem[0].to_string();
            } else if file_clean.is_empty() && all_ident_like {
                symbol_owned = rem.join(",");
            } else if file_clean.is_empty()
                && let Some(recent_first) = crate::load_recent_files(env).into_iter().next()
            {
                file_clean = recent_first;
                pattern_owned = rem.join(" ");
            } else {
                pattern_owned = rem.join(" ");
            }
        }
    }

    if file_clean.is_empty() && crate::looks_like_file_arg(&symbol_owned) {
        file_clean = crate::clean_rel_arg(&symbol_owned);
        symbol_owned.clear();
    }

    symbol_owned = normalize_symbol_query(&symbol_owned);
    if file_clean.is_empty() && crate::looks_like_file_arg(&symbol_owned) {
        file_clean = crate::clean_rel_arg(&symbol_owned);
        symbol_owned.clear();
    }
    if matches!(
        symbol_owned.to_ascii_lowercase().as_str(),
        "*" | "all" | "file" | "outline" | "full"
    ) {
        symbol_owned.clear();
        if file_clean.is_empty()
            && let Some(recent_first) = crate::load_recent_files(env).into_iter().next()
        {
            file_clean = recent_first;
        }
    } else if file_clean.is_empty()
        && matches!(
            symbol_owned.to_ascii_lowercase().as_str(),
            "imports"
                | "import"
                | "header"
                | "top"
                | "head"
                | "module"
                | "globals"
                | "constants"
                | "__all__"
        )
        && let Some(recent_first) = crate::load_recent_files(env).into_iter().next()
    {
        file_clean = recent_first;
    }
    let file = file_clean.as_str();
    let symbol = symbol_owned.as_str();
    let pattern = pattern_owned.as_str();
    let start: Option<usize> = args
        .get("start")
        .or_else(|| args.get("start_line"))
        .and_then(|s| s.parse().ok());
    let end: Option<usize> = args
        .get("end")
        .or_else(|| args.get("end_line"))
        .and_then(|s| s.parse().ok());

    // Multi-symbol comma-separated support (`symbol="ApplicationService,ProjectDef"`)
    if symbol.contains(',') {
        let mut out = String::new();
        for sub_sym in symbol
            .split(',')
            .map(normalize_symbol_query)
            .filter(|s| !s.is_empty())
            .take(4)
        {
            let mut sub_args = BTreeMap::new();
            if !file.is_empty() {
                sub_args.insert("file".to_string(), file.to_string());
            }
            sub_args.insert("symbol".to_string(), sub_sym);
            out.push_str(&run_view(env, &sub_args)?);
            out.push('\n');
        }
        return Ok(clip_tokens(&out, 1250));
    }

    // Case 1: bare `symbol=Foo.bar` without `file=`
    if file.is_empty() && !symbol.is_empty() {
        let index = load_or_build_index(env)?;
        let matches = find_repo_symbols(&index, symbol);
        if matches.is_empty() {
            // Check module-level TypeAlias / constant assignments across non-test source files (e.g. `ServiceDef = Annotated[...]`)
            let mut assign_out = String::new();
            let mut viewed_files = Vec::new();
            for f in &index.files {
                if is_test_path(f) || is_docs_path(f) {
                    continue;
                }
                if let Ok(content) = crate::read_lf(&crate::join_rel(&env.ws, f))
                    && let Some((lbl, s_ln, e_ln)) =
                        crate::patch::find_module_assignment_span(&content, symbol)
                {
                    let total_lines = content.lines().count().max(1);
                    let s = s_ln.saturating_sub(2).max(1);
                    let e = (e_ln + 12).min(total_lines);
                    viewed_files.push(f.clone());
                    assign_out.push_str(&format!(
                        "--- {f}:{s}-{e} (module assignment `{lbl}`, file has {total_lines} lines) ---\n"
                    ));
                    assign_out.push_str(&format_numbered_slice(&content, s, e));
                    if viewed_files.len() >= 2 {
                        break;
                    }
                }
            }
            if !assign_out.is_empty() {
                crate::record_recent_files(env, &viewed_files);
                return Ok(clip_tokens(&assign_out, 1200));
            }
            return Ok(format!(
                "ERROR: symbol `{symbol}` not found in repository AST index. Pass `\"file\": \"pkg/mod.py\"` or run `scripts/debug.py` with `\"query\": \"{symbol}\"`."
            ));
        }
        let mut viewed_files = Vec::new();
        let mut out = String::new();
        for d in matches.into_iter().take(2) {
            if let Ok(content) = crate::read_lf(&crate::join_rel(&env.ws, &d.file)) {
                viewed_files.push(d.file.clone());
                out.push_str(&format_symbol_with_methods(&content, d, 110));
            }
        }
        crate::record_recent_files(env, &viewed_files);
        return Ok(clip_tokens(&out, 1200));
    }

    if file.is_empty() {
        return Ok(
            "USAGE: view (`scripts/view.py`) with `file=\"pkg/mod.py\"` and optional `symbol=\"Class.method\"` or `start=10`, `end=80` (or `targets=\"pkg/a.py:foo,pkg/b.py:1-40\"`)."
                .to_string(),
        );
    }

    let resolved_file =
        crate::resolve_existing_repo_file(env, file).unwrap_or_else(|| file.to_string());
    let file = resolved_file.as_str();
    let full_path = crate::join_rel(&env.ws, file);
    if full_path.is_dir() {
        return list_workspace_files(env, file);
    }
    let Ok(content) = crate::read_lf(&full_path) else {
        return Ok(format!("ERROR: file `{file}` does not exist in workspace."));
    };
    crate::record_recent_files(env, std::slice::from_ref(&resolved_file));
    let total_lines = content.lines().count().max(1);

    // Case 2: `file` + `symbol`
    if !symbol.is_empty() {
        let (defs, _) = parse_python_file(file, &content);
        if let Some(d) = find_best_symbol(&defs, symbol) {
            let mut out = format_symbol_with_methods(&content, d, 130);
            // a property's getter, `@x.setter` and `@x.deleter` (or `@overload`s) share one
            // qualname: show them all
            let mut twins: Vec<&DefNode> = defs
                .iter()
                .filter(|o| {
                    o.qualname == d.qualname && o.start_line != d.start_line && o.kind != "class"
                })
                .collect();
            twins.sort_by_key(|o| o.start_line);
            for o in twins.into_iter().take(3) {
                out.push_str(&format_symbol_with_methods(&content, o, 60));
            }
            return Ok(clip_tokens(&out, 1250));
        }
        // Inherited method lookup (`symbol="SubClass.method"` where `SubClass` exists in `defs` as a `class`, e.g. `RichOption.__init__`)
        if let Some((cls_part, meth_part)) = symbol.rsplit_once('.')
            && let Some(cls_def) = find_best_symbol(&defs, cls_part)
            && cls_def.kind == "class"
        {
            // only claim inheritance when some class really defines the method; a misnamed method
            // gets the closest method names instead
            let defined_anywhere = crate::load_or_build_index(env)
                .map(|idx| idx.nodes.values().any(|n| n.name == meth_part))
                .unwrap_or(true);
            let mut out = if defined_anywhere {
                format!(
                    "NOTE: class `{}` in `{file}`:{}-{} does not override `{meth_part}` directly (inherits `{meth_part}` from a base class). Showing `{}`:\n",
                    cls_def.qualname, cls_def.start_line, cls_def.end_line, cls_def.qualname
                )
            } else {
                let prefix = format!("{}.", cls_def.qualname);
                let closest = defs
                    .iter()
                    .filter(|d| d.qualname.starts_with(&prefix))
                    .min_by_key(|d| crate::patch::line_edit_distance(&d.name, meth_part))
                    .map(|d| format!(" Did you mean `{}`?", d.qualname))
                    .unwrap_or_default();
                format!(
                    "NOTE: `{meth_part}` is not defined anywhere in the repository (not in `{}`, not in a base class).{closest} Showing `{}`:\n",
                    cls_def.qualname, cls_def.qualname
                )
            };
            out.push_str(&format_symbol_with_methods(&content, cls_def, 85));
            let suffix = format!(".{meth_part}");
            if let Some(same_file_base) = defs.iter().find(|d| d.qualname.ends_with(&suffix)) {
                out.push_str(&format!(
                    "--- Base/sibling `{}` in `{file}` ---\n",
                    same_file_base.qualname
                ));
                out.push_str(&format_symbol_with_methods(&content, same_file_base, 85));
            } else if let Ok(index) = load_or_build_index(env) {
                let cls_header = format_numbered_slice(
                    &content,
                    cls_def.start_line,
                    (cls_def.start_line + 4).min(cls_def.end_line),
                );
                if let Some(base_meth) = index
                    .nodes
                    .values()
                    .filter(|d| !is_test_path(&d.file) && !is_docs_path(&d.file))
                    .find(|d| {
                        d.qualname.ends_with(&suffix)
                            && d.qualname
                                .rsplit_once('.')
                                .is_some_and(|(b, _)| cls_header.contains(b))
                    })
                    && let Ok(base_src) = crate::read_lf(&crate::join_rel(&env.ws, &base_meth.file))
                {
                    out.push_str(&format!(
                        "--- Base `{}` in `{}` ---\n",
                        base_meth.qualname, base_meth.file
                    ));
                    out.push_str(&format_symbol_with_methods(&base_src, base_meth, 85));
                }
            }
            out.push_str(&format!(
                "TIP: Edit the base method above if shared, OR add `{}.{meth_part}` via `scripts/patch.py` (`file=\"{file}\"`, `symbol=\"{}.{meth_part}\"`, `body=\"def {meth_part}(self, ...):\\n    ...\"`).\n",
                cls_def.qualname, cls_def.qualname
            ));
            return Ok(clip_tokens(&out, 1250));
        }
        // `symbol="imports"` / `"top"` / `"header"` / `"globals"` -> show top of file (`1..65`)
        if matches!(
            symbol.to_ascii_lowercase().as_str(),
            "imports"
                | "import"
                | "header"
                | "top"
                | "head"
                | "module"
                | "globals"
                | "constants"
                | "__all__"
        ) {
            let e = total_lines.min(65);
            let mut out = format!(
                "--- {file}:1-{e} (module header & imports, file has {total_lines} lines) ---\n"
            );
            out.push_str(&format_numbered_slice(&content, 1, e));
            return Ok(clip_tokens(&out, 1150));
        }
        // Same-file module-level TypeAlias / constant assignment (e.g. `ServiceDef = Annotated[...]`)
        if let Some((lbl, s_ln, e_ln)) = crate::patch::find_module_assignment_span(&content, symbol)
        {
            let s = s_ln.saturating_sub(2).max(1);
            let e = (e_ln + 12).min(total_lines);
            let mut out = format!(
                "--- {file}:{s}-{e} (module assignment `{lbl}`, file has {total_lines} lines) ---\n"
            );
            out.push_str(&format_numbered_slice(&content, s, e));
            return Ok(clip_tokens(&out, 1150));
        }
        // Cross-file AST fallback before noisy substring grep (e.g., `symbol="Option"` in `rich_click.py` -> `RichOption` in `rich_option.py`)
        if symbol.len() >= 3
            && let Ok(index) = load_or_build_index(env)
        {
            let cross = find_repo_symbols(&index, symbol);
            if !cross.is_empty() {
                let mut viewed_files = Vec::new();
                let mut out = format!(
                    "NOTE: symbol `{symbol}` is not defined in `{file}`; showing matching repo AST definition(s):\n"
                );
                for d in cross.into_iter().take(2) {
                    if let Ok(other_src) = crate::read_lf(&crate::join_rel(&env.ws, &d.file)) {
                        viewed_files.push(d.file.clone());
                        out.push_str(&format_symbol_with_methods(&other_src, d, 110));
                    }
                }
                crate::record_recent_files(env, &viewed_files);
                return Ok(clip_tokens(&out, 1250));
            }
        }
        // Fallback: substring search for `symbol` inside `file`
        if let Some(hit) = grep_in_file(file, &content, symbol) {
            return Ok(clip_tokens(&hit, 1100));
        }
        // If the file itself is small (`<= 150` lines) or has 0 AST definitions (stub/__init__/constants file),
        // show its contents directly instead of forcing another round-trip
        if total_lines <= 150 || defs.is_empty() {
            let e = total_lines.min(150);
            let mut out = format!(
                "NOTE: symbol `{symbol}` not found in `{file}` ({total_lines} lines); showing `{file}:1-{e}`:\n"
            );
            out.push_str(&format_numbered_slice(&content, 1, e));
            return Ok(clip_tokens(&out, 1250));
        }
        let avail: Vec<String> = defs.iter().take(35).map(|d| d.qualname.clone()).collect();
        return Ok(format!(
            "ERROR: symbol `{symbol}` not found in `{file}`. Available symbols: {} (or pass `\"symbol\": \"imports\"` for lines 1-65)",
            avail.join(", ")
        ));
    }

    // Case 3: `file` + `pattern`
    if !pattern.is_empty() {
        if let Some(hit) = grep_in_file(file, &content, pattern) {
            return Ok(clip_tokens(&hit, 1100));
        }
        return Ok(format!("No matches for `{pattern}` in `{file}`."));
    }

    // Case 4: `file` + `start` / `end`
    if start.is_some() || end.is_some() {
        let s = start.unwrap_or(1).max(1);
        let e = end.unwrap_or(s + 90).min(s + 140).min(total_lines);
        let mut out = format!("--- {file}:{s}-{e} (of {total_lines} lines) ---\n");
        out.push_str(&format_numbered_slice(&content, s, e));
        return Ok(clip_tokens(&out, 1100));
    }

    // Case 5: `file` alone -> AST outline (up to 90 symbols) + top 150 lines (full file for <= 150-line modules)
    let (defs, _) = parse_python_file(file, &content);
    let mut out = format!("--- {file} ({total_lines} lines) ---\n");
    if !defs.is_empty() {
        out.push_str("SYMBOLS:\n");
        for d in defs.iter().take(90) {
            out.push_str(&format!(
                "  {}:{}..{} ({})\n",
                d.qualname, d.start_line, d.end_line, d.kind
            ));
        }
    }
    let e = total_lines.min(150);
    out.push_str(&format!("--- {file}:1-{e} ---\n"));
    out.push_str(&format_numbered_slice(&content, 1, e));
    Ok(clip_tokens(&out, 1250))
}

fn list_workspace_files(env: &EnvPaths, prefix: &str) -> Result<String> {
    let idx = load_or_build_index(env)?;
    let norm_prefix = prefix.trim_matches('/');
    let mut files: Vec<&String> = idx
        .files
        .iter()
        .filter(|f| !is_test_path(f) && !is_docs_path(f))
        .filter(|f| {
            norm_prefix.is_empty()
                || f.as_str() == norm_prefix
                || f.starts_with(&format!("{norm_prefix}/"))
        })
        .collect();
    files.sort();
    let label = if norm_prefix.is_empty() {
        "/workspace"
    } else {
        norm_prefix
    };
    let mut out = format!(
        "--- Python source files in `{label}` ({} files) ---\n",
        files.len()
    );
    for f in files.into_iter().take(80) {
        out.push_str(&format!("  {f}\n"));
    }
    Ok(clip_tokens(&out, 1100))
}

fn view_one_target_spec(env: &EnvPaths, spec: &str) -> Result<String> {
    let mut sub = BTreeMap::new();
    if let Some((f, rest)) = spec.split_once(':') {
        sub.insert("file".to_string(), f.to_string());
        if let Some((s_str, e_str)) = rest.split_once('-').or_else(|| rest.split_once(".."))
            && let (Ok(s), Ok(e)) = (s_str.parse::<usize>(), e_str.parse::<usize>())
        {
            sub.insert("start".to_string(), s.to_string());
            sub.insert("end".to_string(), e.to_string());
        } else if let Ok(line) = rest.parse::<usize>() {
            sub.insert(
                "start".to_string(),
                line.saturating_sub(10).max(1).to_string(),
            );
            sub.insert("end".to_string(), (line + 35).to_string());
        } else {
            sub.insert("symbol".to_string(), rest.to_string());
        }
    } else {
        sub.insert("file".to_string(), spec.to_string());
    }
    run_view(env, &sub)
}

pub(crate) fn unique_class_methods<'a, I>(methods_iter: I, max_count: usize) -> Vec<String>
where
    I: IntoIterator<Item = &'a DefNode>,
{
    let mut order: Vec<String> = Vec::new();
    let mut best_span: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for m in methods_iter {
        let span = m.end_line.saturating_sub(m.start_line);
        match best_span.get(&m.name) {
            None => {
                order.push(m.name.clone());
                best_span.insert(m.name.clone(), (m.start_line, m.end_line));
            }
            Some(&(prev_s, prev_e)) => {
                if span > prev_e.saturating_sub(prev_s) {
                    best_span.insert(m.name.clone(), (m.start_line, m.end_line));
                }
            }
        }
    }
    order
        .into_iter()
        .take(max_count)
        .filter_map(|nm| best_span.get(&nm).map(|(s, e)| format!("{nm}:{s}..{e}")))
        .collect()
}

fn format_symbol_with_methods(content: &str, d: &DefNode, max_body_lines: usize) -> String {
    let mut out = String::new();
    if d.kind == "class" && d.end_line.saturating_sub(d.start_line) > 85 {
        let (defs, _) = parse_python_file(&d.file, content);
        let prefix = format!("{}.", d.qualname);
        let methods =
            unique_class_methods(defs.iter().filter(|m| m.qualname.starts_with(&prefix)), 120);
        if !methods.is_empty() {
            out.push_str(&format!(
                "METHODS OF `{}` in `{}`: {}\n",
                d.qualname,
                d.file,
                methods.join(", ")
            ));
        }
    }
    out.push_str(&format_def_body_aware(content, d, max_body_lines));
    out
}

fn find_repo_symbols<'a>(
    index: &'a crate::ast_graph::RepoIndex,
    raw_symbol: &str,
) -> Vec<&'a DefNode> {
    let clean = normalize_symbol_query(raw_symbol);
    if clean.is_empty() {
        return Vec::new();
    }
    let clean_low = clean.to_ascii_lowercase();
    let mut exact: Vec<&DefNode> = index
        .nodes
        .values()
        .filter(|d| !is_test_path(&d.file) && !is_docs_path(&d.file))
        .filter(|d| {
            d.qualname == clean
                || d.name == clean
                || d.qualname.ends_with(&format!(".{clean}"))
                || d.qualname.eq_ignore_ascii_case(&clean)
                || d.name.eq_ignore_ascii_case(&clean)
        })
        .collect();
    if !exact.is_empty() {
        exact.sort_by_key(|d| (d.qualname != clean, d.name != clean, d.file.as_str()));
        return exact;
    }
    // Boundary-aware suffix/prefix match on symbol names (e.g., `Option` -> `RichOption`, `BrokenPipe` -> `on_broken_pipe`)
    if clean_low.len() >= 4 {
        let mut fuzzy: Vec<&DefNode> = index
            .nodes
            .values()
            .filter(|d| !is_test_path(&d.file) && !is_docs_path(&d.file))
            .filter(|d| {
                let nl = d.name.to_ascii_lowercase();
                nl.ends_with(&clean_low)
                    || nl.starts_with(&clean_low)
                    || nl.contains(&format!("_{clean_low}"))
            })
            .collect();
        fuzzy.sort_by_key(|d| (d.kind != "class", d.name.len(), d.file.as_str()));
        return fuzzy;
    }
    Vec::new()
}

pub fn find_best_symbol<'a>(defs: &'a [DefNode], symbol: &str) -> Option<&'a DefNode> {
    let norm = normalize_symbol_query(symbol);
    let clean = norm.as_str();
    let best_of_first_qual = |pred: &dyn Fn(&DefNode) -> bool| -> Option<&'a DefNode> {
        let first = defs.iter().find(|d| pred(d))?;
        let target_qual = &first.qualname;
        defs.iter()
            .filter(|d| d.qualname == *target_qual)
            .max_by_key(|d| (d.end_line.saturating_sub(d.start_line), d.start_line))
    };
    best_of_first_qual(&|d| d.qualname == clean)
        .or_else(|| best_of_first_qual(&|d| d.name == clean))
        .or_else(|| {
            let suffix = format!(".{clean}");
            best_of_first_qual(&|d| d.qualname.ends_with(&suffix))
        })
        .or_else(|| {
            best_of_first_qual(&|d| {
                d.qualname.eq_ignore_ascii_case(clean) || d.name.eq_ignore_ascii_case(clean)
            })
        })
}

fn grep_in_file(file: &str, content: &str, pattern: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();
    let pat_low = pattern.to_ascii_lowercase();
    let mut hit_lines = Vec::new();
    for (idx, ln) in lines.iter().enumerate() {
        if ln.contains(pattern) || ln.to_ascii_lowercase().contains(&pat_low) {
            hit_lines.push(idx + 1);
        }
    }
    if hit_lines.is_empty() {
        return None;
    }
    let mut out = format!("--- {file} matches for `{pattern}` ---\n");
    let mut last_end = 0usize;
    for &h in hit_lines.iter().take(4) {
        let s = h.saturating_sub(3).max(1).max(last_end + 1);
        let e = (h + 6).min(lines.len());
        if s <= e {
            out.push_str(&format_numbered_slice(content, s, e));
            last_end = e;
        }
    }
    Some(out)
}
