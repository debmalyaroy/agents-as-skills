//! `/tmp/coder patch` (alias `edit`): single or multi-file/multi-symbol patcher with
//! indent/escape repair, syntax gate, 3-way merge (`git merge-file` + line-delta splice),
//! write-through `/tmp/.agent/best.patch` backup and static critic.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use regex::Regex;

use crate::ast_graph::{
    PYTHON_BUILTINS, Signature, changed_signatures, check_deleted_public_defs,
    check_new_undefined_and_self_methods, check_python_syntax, inside_multiline_string,
    is_test_path, parse_python_file,
};
use crate::view::find_best_symbol;
use crate::{EnvPaths, clip_tokens, load_or_build_index, run_shell_with_timeout};

#[derive(Debug, Clone, Default)]
pub struct PatchBlock {
    pub file: String,
    pub symbol: Option<String>,
    pub after_symbol: Option<String>,
    pub old: Option<String>,
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub after: Option<usize>,
    pub imports: Option<String>,
    pub create: bool,
    pub append: bool,
    pub delete: bool,
    pub force: bool,
    pub new_text: String,
}

/// Exclusive file lock guard in `/tmp/.agent/merge.lock` so concurrent `/tmp/coder patch` calls
/// serialize cleanly and 3-way merge without race conditions.
pub struct MergeLockGuard {
    lock_path: std::path::PathBuf,
}

impl MergeLockGuard {
    pub fn acquire(agent_dir: &Path) -> Self {
        let _ = std::fs::create_dir_all(agent_dir);
        let lock_path = agent_dir.join("merge.lock");
        for _ in 0..200 {
            if OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
                .is_ok()
            {
                return Self { lock_path };
            }
            // Break stale locks older than 30 seconds
            if let Ok(meta) = std::fs::metadata(&lock_path)
                && let Ok(el) = meta
                    .modified()
                    .and_then(|m| m.elapsed().map_err(std::io::Error::other))
                && el > Duration::from_secs(30)
            {
                let _ = std::fs::remove_file(&lock_path);
            }
            thread::sleep(Duration::from_millis(25));
        }
        Self { lock_path }
    }
}

impl Drop for MergeLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

fn strip_outer_markdown_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let mut lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() >= 2
            && lines
                .first()
                .is_some_and(|l| l.trim_start().starts_with("```"))
            && lines.last().is_some_and(|l| l.trim() == "```")
        {
            lines.remove(0);
            lines.pop();
            let mut inner = lines.join("\n");
            inner.push('\n');
            return inner;
        }
    }
    raw.to_string()
}

/// Parse unified diff hunks (`--- a/pkg/mod.py` / `+++ b/pkg/mod.py` / `@@ ... @@`) into `PatchBlock`s.
pub(crate) fn try_parse_unified_diff(default_file: &str, body: &str) -> Option<Vec<PatchBlock>> {
    if !body.contains("@@ -") && !body.contains("@@\n") {
        return None;
    }
    let mut blocks = Vec::new();
    let mut cur_file = default_file.to_string();
    let mut in_hunk = false;
    let mut old_lines: Vec<String> = Vec::new();
    let mut new_lines: Vec<String> = Vec::new();

    let flush_hunk = |file: &str,
                      old_l: &mut Vec<String>,
                      new_l: &mut Vec<String>,
                      out: &mut Vec<PatchBlock>| {
        if !old_l.is_empty() || !new_l.is_empty() {
            out.push(PatchBlock {
                file: file.to_string(),
                old: (!old_l.is_empty()).then(|| old_l.join("\n")),
                new_text: new_l.join("\n"),
                ..Default::default()
            });
            old_l.clear();
            new_l.clear();
        }
    };

    for raw_line in body.replace("\r\n", "\n").lines() {
        if let Some(rest) = raw_line.strip_prefix("+++ ") {
            flush_hunk(&cur_file, &mut old_lines, &mut new_lines, &mut blocks);
            in_hunk = false;
            let p = rest.split_whitespace().next().unwrap_or(rest).trim();
            if p != "/dev/null" {
                let stripped = p.strip_prefix("b/").unwrap_or(p);
                // `+++ src/pkg/mod.py:621`: a line number glued to the path,
                // or a range copied from a `view` header (`pkg/core.py:676-693 (function ..)`)
                let stripped = match stripped.rsplit_once(':') {
                    Some((path, line))
                        if !line.is_empty()
                            && line.bytes().all(|b| b.is_ascii_digit() || b == b'-') =>
                    {
                        path
                    }
                    _ => stripped,
                };
                cur_file = crate::clean_rel_arg(stripped);
            }
            continue;
        }
        if raw_line.starts_with("--- ")
            || raw_line.starts_with("diff --git ")
            || raw_line.starts_with("index ")
        {
            if in_hunk {
                flush_hunk(&cur_file, &mut old_lines, &mut new_lines, &mut blocks);
                in_hunk = false;
            }
            continue;
        }
        if raw_line.starts_with("@@ ") {
            flush_hunk(&cur_file, &mut old_lines, &mut new_lines, &mut blocks);
            in_hunk = true;
            continue;
        }
        if !in_hunk {
            continue;
        }
        if raw_line.starts_with("\\ No newline") {
            continue;
        }
        // a `>>>` marker of the <<<OLD/===/>>> form closing a unified diff is dropped, not read as
        // a context line
        let bare = raw_line.trim();
        if bare == ">>>" || bare == "===" || bare.starts_with("<<<OLD") {
            continue;
        }
        if let Some(added) = raw_line.strip_prefix('+') {
            new_lines.push(added.to_string());
        } else if let Some(removed) = raw_line.strip_prefix('-') {
            old_lines.push(removed.to_string());
        } else {
            let ctx = raw_line.strip_prefix(' ').unwrap_or(raw_line);
            old_lines.push(ctx.to_string());
            new_lines.push(ctx.to_string());
        }
    }
    flush_hunk(&cur_file, &mut old_lines, &mut new_lines, &mut blocks);
    (!blocks.is_empty()).then_some(blocks)
}

/// Drop a last line that is only an indented heredoc terminator (` EOF`): bash ends a heredoc only
/// at an exact `EOF` line, so an indented one is read as text.
/// The unique window of `src` whose non-blank lines equal `old`'s non-blank lines (trimmed): `old` copied
/// with a blank line dropped or added.
/// Returns the file's own window, blank lines included; `None` unless exactly one.
pub fn blankless_realign(src: &str, old: &str) -> Option<String> {
    let want: Vec<&str> = old
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if want.len() < 2 {
        return None;
    }
    let lines: Vec<&str> = src.lines().collect();
    let mut hits = Vec::new();
    for start in 0..lines.len() {
        if lines[start].trim() != want[0] {
            continue;
        }
        let (mut i, mut k) = (start, 0usize);
        while i < lines.len() && k < want.len() {
            let t = lines[i].trim();
            if t.is_empty() {
                i += 1;
                continue;
            }
            if t != want[k] {
                break;
            }
            i += 1;
            k += 1;
        }
        if k == want.len() {
            hits.push(lines[start..i].join("\n"));
        }
    }
    (hits.len() == 1).then(|| hits.remove(0))
}

/// Cut a body at leaked chat-template tokens (`<tool_call|>`, `<|channel>`, `<|tool_call>`, `<|"|>`):
/// text generated past the end of the code is not part of the patch.
/// A code fence right before the cut goes too. These tokens never occur in code.
pub fn cut_template_leak(body: &str) -> String {
    const MARKS: [&str; 4] = ["<tool_call|>", "<|channel>", "<|tool_call>", "<|\"|>"];
    let Some(pos) = MARKS.iter().filter_map(|m| body.find(m)).min() else {
        return body.to_string();
    };
    let head = body[..pos].trim_end_matches(['}', ' ', '\t']);
    let head = head.strip_suffix("```").unwrap_or(head);
    let mut out = head.trim_end_matches(['\n', '\r', ' ']).to_string();
    out.push('\n');
    out
}

pub fn strip_stray_eof(body: &str) -> String {
    let trimmed = body.trim_end_matches(['\n', '\r']);
    match trimmed.rsplit_once('\n') {
        Some((head, last)) if last.trim() == "EOF" => format!("{head}\n"),
        None if trimmed.trim() == "EOF" => String::new(),
        _ => body.to_string(),
    }
}

/// Parse single or multi-block batch (`=== file=... symbol=... ===`) patch input.
pub fn parse_patch_blocks(args: &BTreeMap<String, String>, raw_body: &str) -> Vec<PatchBlock> {
    let body_owned = strip_outer_markdown_fence(&strip_stray_eof(&cut_template_leak(raw_body)));
    let body = body_owned.as_str();
    // the header must carry a `key=value` (`=== file=a.py symbol=f ===`): a bare `=======` separator
    // (`<<<<<<< OLD / ======= / >>>>>>>`) is not a header
    let header_re = Regex::new(r"(?m)^===\s*([A-Za-z_]+=.+?)\s*===[ \t]*\r?\n?").expect("regex");
    let mut blocks = Vec::new();

    if header_re.is_match(body) {
        let matches: Vec<_> = header_re.captures_iter(body).collect();
        let spans: Vec<_> = header_re.find_iter(body).collect();
        for i in 0..spans.len() {
            let hdr_str = matches[i].get(1).map(|m| m.as_str()).unwrap_or("");
            let content_start = spans[i].end();
            let content_end = if i + 1 < spans.len() {
                spans[i + 1].start()
            } else {
                body.len()
            };
            let section_body = &body[content_start..content_end];
            let hdr_args = crate::parse_kv_words(hdr_str);
            blocks.extend(expand_old_new_markers(&hdr_args, section_body));
        }
        return blocks;
    }

    // Support inline `file=... [symbol=...]` on line 1 of `body` when `args` omitted `file=`
    if !args.contains_key("file")
        && !args.contains_key("path")
        && !args.contains_key("filepath")
        && !args.contains_key("file_path")
        && let Some((first_line, rest_body)) = body.split_once('\n')
    {
        let ft = first_line.trim();
        if (ft.starts_with("file=") || ft.starts_with("path=") || ft.starts_with("filepath="))
            && !ft.contains('(')
        {
            let mut merged_args = args.clone();
            for (k, v) in crate::parse_kv_words(ft) {
                merged_args.entry(k).or_insert(v);
            }
            return expand_old_new_markers(&merged_args, rest_body);
        }
    }

    let default_file = args
        .get("file")
        .or_else(|| args.get("path"))
        .or_else(|| args.get("filepath"))
        .or_else(|| args.get("file_path"))
        .map(|s| crate::clean_rel_arg(s))
        .unwrap_or_default();
    if let Some(mut diff_blocks) = try_parse_unified_diff(&default_file, body) {
        // keep the call's `symbol=`: it disambiguates a hunk whose context occurs twice in the file
        let symbol = args
            .get("symbol")
            .or_else(|| args.get("name"))
            .map(|s| crate::view::normalize_symbol_query(s))
            .filter(|s| !s.is_empty());
        if symbol.is_some() {
            for b in &mut diff_blocks {
                if b.symbol.is_none() {
                    b.symbol = symbol.clone();
                }
            }
        }
        return diff_blocks;
    }

    expand_old_new_markers(args, body)
}

/// A unified-diff hunk (`@@`/`---`/`+++` headers optional) as (old, new) text: `-` and context lines
/// make the old text, `+` and context lines the new. `None` unless the body has at least one `-` or
/// `+` line and every non-empty line is a hunk line (so plain code that happens to start with `-` is
/// never misread).
pub fn unified_hunk_to_old_new(body: &str) -> Option<(String, String)> {
    let mut old = Vec::new();
    let mut new = Vec::new();
    let mut changed = 0usize;
    for line in body.lines() {
        if line.starts_with("@@") || line.starts_with("--- ") || line.starts_with("+++ ") {
            continue;
        }
        if let Some(r) = line.strip_prefix('-') {
            old.push(r.to_string());
            changed += 1;
        } else if let Some(r) = line.strip_prefix('+') {
            new.push(r.to_string());
            changed += 1;
        } else if let Some(r) = line.strip_prefix(' ') {
            // context: in both (a hunk's context lines carry one leading space)
            old.push(r.to_string());
            new.push(r.to_string());
        } else if line.trim().is_empty() {
            old.push(String::new());
            new.push(String::new());
        } else {
            return None;
        }
    }
    (changed > 0 && !old.iter().all(|l| l.trim().is_empty()))
        .then(|| (old.join("\n"), new.join("\n")))
}

/// `--- path:A-B` (+ optional `+++ path:..`) headers followed only by replacement lines, no `@@` and
/// no `-`/`+` lines: "replace lines A..B with this".
/// Returns (path, A, B, new text).
pub fn header_range_body(body: &str) -> Option<(String, usize, usize, String)> {
    let re = Regex::new(r"^--- (?:a/)?(\S+?):(\d+)-(\d+)\b").expect("regex");
    let mut lines = body.lines();
    let c = re.captures(lines.next()?)?;
    let (start, end): (usize, usize) = (c[2].parse().ok()?, c[3].parse().ok()?);
    if start == 0 || end < start {
        return None;
    }
    let rest: Vec<&str> = lines.collect();
    let rest: &[&str] = match rest.first() {
        Some(l) if l.starts_with("+++ ") => &rest[1..],
        _ => &rest[..],
    };
    let is_hunk = rest.iter().any(|l| {
        l.starts_with("@@") || (l.starts_with('-') && !l.starts_with("---")) || l.starts_with('+')
    });
    // `--- path:A-B` then old / === / new / >>> is the separator form with a copied header
    let has_markers = rest.iter().any(|l| {
        let t = l.trim_end();
        ((3..=7).contains(&t.len()) && t.chars().all(|c| c == '='))
            || t.trim_start().starts_with("<<<")
            || t.trim_start().starts_with(">>>")
    });
    if is_hunk || has_markers || rest.iter().all(|l| l.trim().is_empty()) {
        return None;
    }
    Some((c[1].to_string(), start, end, rest.join("\n")))
}

/// `old lines / === / new lines` without the `<<<OLD` and `>>>` markers.
/// Exactly one `===` line, non-empty old text.
pub fn bare_separator_old_new(body: &str) -> Option<(String, String)> {
    let mut lines: Vec<&str> = body.lines().collect();
    // only one of the two markers written: `old / === / new / >>>`
    // or `<<<OLD / old / === / new`
    if lines.last().is_some_and(|l| {
        let t = l.trim();
        t.len() >= 3 && t.chars().all(|c| c == '>')
    }) {
        lines.pop();
    }
    if lines
        .first()
        .is_some_and(|l| l.trim_start().starts_with("<<<"))
    {
        lines.remove(0);
    }
    // closed with a second `===` instead of `>>>`: the last separator is the end
    let is_sep = |l: &str| {
        let t = l.trim_end();
        (3..=7).contains(&t.len()) && t.chars().all(|c| c == '=')
    };
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    if lines.iter().filter(|l| is_sep(l)).count() == 2 && lines.last().is_some_and(|l| is_sep(l)) {
        lines.pop();
    }
    let seps: Vec<usize> = (0..lines.len())
        // at column 0 and short: an indented docstring underline (`========`) is not a separator
        .filter(|&i| {
            let t = lines[i].trim_end();
            (3..=7).contains(&t.len()) && t.chars().all(|c| c == '=')
        })
        .collect();
    let [k] = seps[..] else { return None };
    let old = lines[..k].join("\n");
    let new = lines[k + 1..].join("\n");
    (!old.trim().is_empty()).then_some((old, new))
}

/// Support `<<<OLD ... === ... >>>` (or `<<<<<<< SEARCH ... ======= ... >>>>>>>`) inside a block body.
fn expand_old_new_markers(args: &BTreeMap<String, String>, raw_body: &str) -> Vec<PatchBlock> {
    let file = args
        .get("file")
        .or_else(|| args.get("path"))
        .or_else(|| args.get("filepath"))
        .or_else(|| args.get("file_path"))
        .map(|s| crate::clean_rel_arg(s))
        .unwrap_or_default();
    let norm_symbol = args
        .get("symbol")
        .or_else(|| args.get("name"))
        .map(|s| crate::view::normalize_symbol_query(s))
        .filter(|s| !s.is_empty());
    let norm_after_symbol = args
        .get("after_symbol")
        .map(|s| crate::view::normalize_symbol_query(s))
        .filter(|s| !s.is_empty());
    let start_line = args
        .get("start")
        .or_else(|| args.get("start_line"))
        .and_then(|s| s.parse().ok());
    let end_line = args
        .get("end")
        .or_else(|| args.get("end_line"))
        .and_then(|s| s.parse().ok());
    let force_flag = args
        .get("force")
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.is_empty());
    let delete_flag = args
        .get("delete")
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

    let search_re = Regex::new(
        r"(?ms)^[ \t]*<<<+[ \t]*(?:OLD|SEARCH)[ \t]*\r?\n(.*?)\r?\n[ \t]*===+[ \t]*\r?\n(.*?)\r?\n[ \t]*>>>+[ \t]*(?:NEW|REPLACE)?[ \t]*$",
    )
    .expect("regex");

    if search_re.is_match(raw_body) {
        let mut out = Vec::new();
        for cap in search_re.captures_iter(raw_body) {
            out.push(PatchBlock {
                file: file.clone(),
                symbol: norm_symbol.clone(),
                start: start_line,
                end: end_line,
                force: force_flag,
                delete: delete_flag,
                old: Some(cap[1].to_string()),
                new_text: cap[2].to_string(),
                ..Default::default()
            });
        }
        if !out.is_empty() {
            return out;
        }
    }

    // Flat call shape: a heredoc written as a unified-diff hunk. A body whose changed lines all
    // carry `-`/`+` becomes old/new text.
    if !args.contains_key("old")
        && !args.contains_key("old_string")
        && let Some((path, start, end, new)) = header_range_body(raw_body)
    {
        return vec![PatchBlock {
            file: if path.is_empty() {
                file
            } else {
                crate::clean_rel_arg(&path)
            },
            start: Some(start),
            end: Some(end),
            force: force_flag,
            new_text: new,
            ..Default::default()
        }];
    }
    if !args.contains_key("old")
        && !args.contains_key("old_string")
        && let Some((old, new)) =
            unified_hunk_to_old_new(raw_body).or_else(|| bare_separator_old_new(raw_body))
    {
        return vec![PatchBlock {
            file,
            symbol: norm_symbol,
            start: start_line,
            end: end_line,
            force: force_flag,
            delete: delete_flag,
            old: Some(old),
            new_text: new,
            ..Default::default()
        }];
    }

    let old_val = args
        .get("old")
        .or_else(|| args.get("old_string"))
        .or_else(|| args.get("old_str"))
        .or_else(|| args.get("snippet"))
        .or_else(|| args.get("search"))
        .or_else(|| args.get("target"))
        .or_else(|| args.get("original"))
        .cloned();

    let new_text = args
        .get("new")
        .or_else(|| args.get("new_string"))
        .or_else(|| args.get("new_str"))
        .or_else(|| args.get("replacement"))
        .or_else(|| args.get("replace"))
        .cloned()
        .unwrap_or_else(|| raw_body.to_string());

    vec![PatchBlock {
        file,
        symbol: norm_symbol,
        after_symbol: norm_after_symbol,
        old: old_val,
        start: start_line,
        end: end_line,
        after: args.get("after").and_then(|s| s.parse().ok()),
        imports: args.get("imports").cloned().or_else(|| {
            (args.contains_key("imports") && !raw_body.is_empty()).then(|| raw_body.to_string())
        }),
        create: args
            .get("create")
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.is_empty()),
        append: args
            .get("append")
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.is_empty()),
        delete: delete_flag,
        force: force_flag,
        new_text,
    }]
}

/// `scripts/{debug,view,patch,verify,probe}.py`: the swe-coder entry points the model calls.
pub fn is_skill_script_path(rel_file: &str) -> bool {
    let norm = rel_file.trim_start_matches("./").replace('\\', "/");
    matches!(
        norm.as_str(),
        "scripts/debug.py"
            | "scripts/view.py"
            | "scripts/patch.py"
            | "scripts/verify.py"
            | "scripts/probe.py"
    )
}

pub fn is_protected_file(rel_file: &str, issue_text: &str) -> bool {
    let base = Path::new(rel_file)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if matches!(
        base,
        "conftest.py" | "pytest.ini" | "tox.ini" | "setup.cfg" | "pyproject.toml"
    ) && !issue_text.contains(base)
    {
        return true;
    }
    is_test_path(rel_file) && !issue_text.contains(rel_file)
}

/// Remove `view` line-number prefixes (`"   403  code"`) if the model copied them into `old` or `new`.
pub fn strip_gutter(text: &str) -> (String, bool) {
    // a `view` header copied as the first line (`--- pkg/core.py:677-694 (function ..) ---`)
    let header_re = Regex::new(r"^--- \S+:\d+(?:-\d+)?(?: \(.*\))?(?: ---)?\s*$").expect("regex");
    let owned;
    let text = match text.split_once('\n') {
        Some((first, rest)) if header_re.is_match(first) => {
            owned = rest.to_string();
            owned.as_str()
        }
        _ => text,
    };
    let gutter_re = Regex::new(r"^ {0,8}\d{1,6}(?::| {2}|\t)").expect("regex");
    let non_empty: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if non_empty.is_empty() || !non_empty.iter().all(|l| gutter_re.is_match(l)) {
        return (text.to_string(), false);
    }
    let cleaned: Vec<String> = text
        .lines()
        .map(|l| {
            if l.trim().is_empty() {
                String::new()
            } else {
                gutter_re.replace(l, "").into_owned()
            }
        })
        .collect();
    let mut res = cleaned.join("\n");
    if text.ends_with('\n') {
        res.push('\n');
    }
    (res, true)
}

pub fn unescape_once(s: &str) -> String {
    s.replace("\\\r\n", "\n")
        .replace("\\\n", "\n")
        .replace("\\\"", "\"")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
}

fn line_indent(line: &str) -> isize {
    (line.len() - line.trim_start_matches(' ').len()) as isize
}

fn shift_indent(text: &str, delta: isize) -> String {
    let mut out = String::new();
    for ln in text.split_inclusive('\n') {
        if ln.trim().is_empty() {
            out.push_str(ln);
        } else {
            let cur = line_indent(ln);
            let new_ind = (cur + delta).max(0) as usize;
            out.push_str(&" ".repeat(new_ind));
            out.push_str(ln.trim_start_matches(' '));
        }
    }
    out
}

/// Try compiling `assembled`; if it has a Python syntax/indentation error, try unescaping and shifting
/// indentation (`-12..=+12` spaces) of `new_block` via `assemble_fn(candidate_block)` until it compiles
fn auto_repair_and_validate<F>(
    initial_assembled: String,
    new_block: &str,
    target_indent: Option<isize>,
    mut assemble_fn: F,
) -> std::result::Result<(String, String), String>
where
    F: FnMut(&str) -> String,
{
    if check_python_syntax(&initial_assembled).is_none() {
        return Ok((initial_assembled, String::new()));
    }

    let mut base_variants: Vec<(String, String)> = vec![(new_block.to_string(), String::new())];
    let unesc = unescape_once(new_block);
    if unesc != new_block {
        base_variants.insert(
            0,
            (unesc, "unescaped literal \\n/\\\" in patch".to_string()),
        );
    }

    // Compute current minimum non-empty line indentation of `new_block`
    let mut shifts: Vec<isize> = vec![0];
    if let Some(want) = target_indent {
        let min_have = new_block
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(line_indent)
            .min()
            .unwrap_or(0);
        let d = want - min_have;
        if d != 0 {
            shifts.push(d);
        }
    }
    for d in [-4, 4, -8, 8, -2, 2, -12, 12] {
        if !shifts.contains(&d) {
            shifts.push(d);
        }
    }

    for (variant_txt, note_prefix) in &base_variants {
        for &d in &shifts {
            if d == 0 && note_prefix.is_empty() {
                continue;
            }
            let shifted = if d == 0 {
                variant_txt.clone()
            } else {
                shift_indent(variant_txt, d)
            };
            let cand = assemble_fn(&shifted);
            if check_python_syntax(&cand).is_none() {
                let note = match (note_prefix.is_empty(), d == 0) {
                    (true, false) => format!("re-indented block by {d:+} spaces"),
                    (false, true) => note_prefix.clone(),
                    (false, false) => format!("{note_prefix}, re-indented by {d:+} spaces"),
                    (true, true) => String::new(),
                };
                return Ok((cand, note));
            }
        }
        // Also try shifting only lines 2..N if line 1 was written at column 0 in `old` replacement
        let rows: Vec<&str> = variant_txt.split_inclusive('\n').collect();
        if rows.len() > 1
            && let Some(want) = target_indent
            && line_indent(rows[0]) == 0
        {
            let rest = rows[1..].concat();
            let min_rest = rest
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(line_indent)
                .min()
                .unwrap_or(0);
            for d in [want - min_rest, want] {
                if d != 0 {
                    let cand_block = format!("{}{}", rows[0], shift_indent(&rest, d));
                    let cand = assemble_fn(&cand_block);
                    if check_python_syntax(&cand).is_none() {
                        return Ok((cand, format!("indented continuation lines by {d:+} spaces")));
                    }
                }
            }
        }
    }

    let (err_line, err_snip) =
        check_python_syntax(&initial_assembled).unwrap_or((1, "syntax error".to_string()));
    let all_lines: Vec<&str> = initial_assembled.lines().collect();
    let ctx_s = err_line.saturating_sub(3).max(1);
    let ctx_e = (err_line + 3).min(all_lines.len().max(1));
    let mut ctx = String::new();
    if !all_lines.is_empty() {
        for ln_no in ctx_s..=ctx_e {
            let marker = if ln_no == err_line { ">>" } else { "  " };
            let txt = all_lines.get(ln_no - 1).copied().unwrap_or("");
            ctx.push_str(&format!("\n    {marker} {ln_no:4}: {txt}"));
        }
    }
    Err(format!(
        "SyntaxError near line {err_line} ({err_snip}): file kept unchanged.{ctx}"
    ))
}

/// Ensure the base snapshot `/tmp/.agent/base/<rel_file>` exists before modifying `/workspace/<rel_file>`.
pub fn ensure_base_snapshot(env: &EnvPaths, rel_file: &str) {
    let base_path = crate::join_rel(&env.agent_dir.join("base"), rel_file);
    if base_path.is_file() {
        return;
    }
    let ws_file = crate::join_rel(&env.ws, rel_file);
    if let Ok(content) = crate::read_lf(&ws_file) {
        if let Some(p) = base_path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let _ = crate::write_lf(&base_path, &content);
    }
}

/// 3-Way File Merge (`git merge-file -p <live> <base> <staged>`) when a line-range edit (`start..end`)
/// was specified against the original base file line numbers after the live file had already been modified
fn three_way_merge_strings(
    env: &EnvPaths,
    rel_file: &str,
    live_src: &str,
    base_src: &str,
    staged_src: &str,
) -> Option<String> {
    if live_src == base_src {
        return Some(staged_src.to_string());
    }
    let merge_dir = env.agent_dir.join("merge_tmp");
    let _ = std::fs::create_dir_all(&merge_dir);
    let safe_name = crate::path_segments(rel_file).join("_");
    let live_p = merge_dir.join(format!("{safe_name}.live"));
    let base_p = merge_dir.join(format!("{safe_name}.base"));
    let staged_p = merge_dir.join(format!("{safe_name}.staged"));

    let _ = crate::write_lf(&live_p, live_src);
    let _ = crate::write_lf(&base_p, base_src);
    let _ = crate::write_lf(&staged_p, staged_src);

    if let Ok(out) = Command::new("git")
        .args([
            "merge-file",
            "-p",
            &live_p.to_string_lossy(),
            &base_p.to_string_lossy(),
            &staged_p.to_string_lossy(),
        ])
        .output()
        && out.status.success()
    {
        let merged = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        if !merged.contains("<<<<<<<") && check_python_syntax(&merged).is_none() {
            return Some(merged);
        }
    }
    None
}

fn dedent_block(text: &str) -> String {
    let min_ind = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(line_indent)
        .min()
        .unwrap_or(0);
    if min_ind > 0 {
        shift_indent(text, -min_ind)
    } else {
        text.to_string()
    }
}

fn align_block_to_indent(text: &str, target_indent: Option<isize>) -> String {
    let Some(want) = target_indent else {
        return text.to_string();
    };
    let Some(min_have) = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(line_indent)
        .min()
    else {
        return text.to_string();
    };
    let delta = want - min_have;
    if delta != 0 {
        shift_indent(text, delta)
    } else {
        text.to_string()
    }
}

fn has_stray_patch_marker(text: &str) -> Option<String> {
    for ln in text.lines() {
        let t = ln.trim();
        if t.starts_with("<<<OLD")
            || t.starts_with("<<<<<<<")
            || t.starts_with(">>>>>>>")
            || t == "==="
            || t == ">>>"
        {
            return Some(t.to_string());
        }
    }
    None
}

fn has_elision_placeholder(text: &str) -> Option<String> {
    let elision_re = Regex::new(
        r"(?i)^\s*#\s*(?:\.\.\.|rest of (?:class|function|method|file|code|implementation)|existing (?:code|methods|implementation)|unchanged|same as (?:before|above)|keep existing)\b",
    )
    .expect("regex");
    for ln in text.lines() {
        if elision_re.is_match(ln) {
            return Some(ln.trim().to_string());
        }
    }
    None
}

/// Auto-resolve `block.file` when omitted or given as a short suffix, and reject non-existent files
/// unless `block.create` is explicitly `true`.
fn resolve_patch_block_file(env: &EnvPaths, block: &mut PatchBlock) -> Result<()> {
    let raw_file = crate::clean_rel_arg(&block.file);
    if !raw_file.is_empty() {
        if block.create {
            if !crate::looks_like_file_arg(&raw_file) {
                bail!("refused to create invalid file path `{raw_file}`");
            }
            block.file = raw_file;
            return Ok(());
        }
        if let Some(resolved) = crate::resolve_existing_repo_file(env, &raw_file) {
            block.file = resolved;
            return Ok(());
        }
        bail!(
            "file `{raw_file}` does not exist in `/workspace` (pass `create=1` if you intend to create a new file, or check the path with `debug`)"
        );
    }

    // `block.file` is empty: try auto-discovering the target file from `old` snippet or `symbol`
    let recent = crate::load_recent_files(env);

    // 1. Auto-discover from `old` snippet (`<<<OLD ... === ... >>>` or `old="..."`)
    if let Some(raw_old) = block.old.as_deref().filter(|s| !s.trim().is_empty()) {
        let (clean_old, _) = strip_gutter(raw_old);
        let u_old = unescape_once(&clean_old);
        let matches_in = |files: &[String]| -> Vec<String> {
            let mut hits = Vec::new();
            for f in files {
                if let Ok(src) = crate::read_lf(&crate::join_rel(&env.ws, f)) {
                    let ok = match_old_snippet(&src, &clean_old).is_some()
                        || (u_old != clean_old && match_old_snippet(&src, &u_old).is_some());
                    if ok && !hits.contains(f) {
                        hits.push(f.clone());
                    }
                }
            }
            hits
        };
        let recent_hits = matches_in(&recent);
        if recent_hits.len() == 1 {
            block.file = recent_hits[0].clone();
            return Ok(());
        }
        if let Ok(idx) = load_or_build_index(env) {
            let non_test_files: Vec<String> = idx
                .files
                .iter()
                .filter(|f| !is_test_path(f))
                .cloned()
                .collect();
            let repo_hits = matches_in(&non_test_files);
            if repo_hits.len() == 1 {
                block.file = repo_hits[0].clone();
                return Ok(());
            } else if repo_hits.len() > 1 {
                bail!(
                    "missing `file=<path>` and `old` snippet matched multiple files ({}) — pass `file=<path>` explicitly",
                    repo_hits.into_iter().take(4).collect::<Vec<_>>().join(", ")
                );
            }
        }
        bail!(
            "missing `file=<path>` and `old` snippet did not match any recently inspected file — pass `file=<path>`"
        );
    }

    // 2. Auto-discover from `after_symbol` first (guaranteed existing anchor), then `symbol`
    if let Some(sym) = block
        .after_symbol
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| block.symbol.as_deref().filter(|s| !s.trim().is_empty()))
    {
        let mut recent_hits = Vec::new();
        for f in &recent {
            if let Ok(src) = crate::read_lf(&crate::join_rel(&env.ws, f)) {
                let (defs, _) = parse_python_file(f, &src);
                if find_best_symbol(&defs, sym).is_some() && !recent_hits.contains(f) {
                    recent_hits.push(f.clone());
                }
            }
        }
        if recent_hits.len() == 1 {
            block.file = recent_hits[0].clone();
            return Ok(());
        }
        if let Ok(idx) = load_or_build_index(env) {
            let mut repo_files = Vec::new();
            for d in idx.nodes.values() {
                if !is_test_path(&d.file)
                    && (d.qualname == sym
                        || d.name == sym
                        || d.qualname.ends_with(&format!(".{sym}")))
                    && !repo_files.contains(&d.file)
                {
                    repo_files.push(d.file.clone());
                }
            }
            if repo_files.len() == 1 {
                block.file = repo_files[0].clone();
                return Ok(());
            } else if repo_files.len() > 1 {
                bail!(
                    "missing `file=<path>` and symbol `{sym}` exists in multiple files ({}) — pass `file=<path>`",
                    repo_files
                        .into_iter()
                        .take(4)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            // New method on an existing class (`symbol="ClassName.new_method"` without `file=`):
            // resolve `ClassName` from recently inspected files first, then repo-wide non-test classes
            if let Some((cls_part, _)) = sym.rsplit_once('.') {
                let mut cls_recent = Vec::new();
                for f in &recent {
                    if let Ok(src) = crate::read_lf(&crate::join_rel(&env.ws, f)) {
                        let (defs, _) = parse_python_file(f, &src);
                        if find_best_symbol(&defs, cls_part).is_some_and(|d| d.kind == "class")
                            && !cls_recent.contains(f)
                        {
                            cls_recent.push(f.clone());
                        }
                    }
                }
                if cls_recent.len() == 1
                    || (cls_recent.len() > 1 && recent.first() == cls_recent.first())
                {
                    block.file = cls_recent[0].clone();
                    return Ok(());
                }
                let mut cls_repo = Vec::new();
                for d in idx.nodes.values() {
                    if !is_test_path(&d.file)
                        && d.kind == "class"
                        && (d.qualname == cls_part || d.name == cls_part)
                        && !cls_repo.contains(&d.file)
                    {
                        cls_repo.push(d.file.clone());
                    }
                }
                if cls_repo.len() == 1 {
                    block.file = cls_repo[0].clone();
                    return Ok(());
                }
            }
        }
        bail!("missing `file=<path>` and symbol `{sym}` was not found in repository index");
    }

    let hint = recent
        .first()
        .map(|f| format!(" (e.g. `file=\"{f}\"`)."))
        .unwrap_or_default();
    bail!("missing `file=<path>` in patch call{hint}");
}

/// Apply one `PatchBlock` to the workspace under the active `MergeLockGuard`.
fn apply_single_block(
    env: &EnvPaths,
    raw_block: &PatchBlock,
    issue_text: &str,
) -> Result<(String, String)> {
    let mut resolved_block = raw_block.clone();
    resolve_patch_block_file(env, &mut resolved_block)?;
    let block = &resolved_block;
    let rel_clean = crate::clean_rel_arg(&block.file);
    let rel_file = rel_clean.as_str();
    if rel_file.is_empty() {
        bail!("missing `file=<path>` in patch block");
    }
    if is_protected_file(rel_file, issue_text) {
        // A protected file (tests, project configuration) is never edited: say so and point back
        // at the code behaviour asked for.
        bail!(
            "refused to edit `{rel_file}`: test files and runner/packaging configs are read-only and are NOT graded. Do NOT retry this edit. The graded fix is a change to the library's source code for the behaviour the issue's TITLE describes: call `scripts/debug.py` with the title's words as `query`"
        );
    }

    let full_path = crate::join_rel(&env.ws, rel_file);
    // the skill's own entry points are not repository files
    if is_skill_script_path(rel_file) && !full_path.is_file() {
        // "patch scripts/verify.py" is an attempt to change the repro: keep the code as the repro
        // instead of only refusing
        let code = block.new_text.trim();
        if rel_file.ends_with("verify.py")
            && !code.is_empty()
            && crate::ast_graph::check_python_syntax(code).is_none()
        {
            let _ = crate::write_lf(&env.agent_dir.join("repro.py"), &format!("{code}\n"));
            bail!(
                "`{rel_file}` is the tool, not a repository file, so nothing was patched; your code is now saved as the repro. Run it: `run_skill_script` with `args={{}}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"` (or pass new code as `args={{\"repro\": \"...\"}}`)"
            );
        }
        bail!(
            "refused to edit `{rel_file}`: `scripts/<name>.py` is the tool you call (`file_path`), not a repository file. Put the repository file to change in `args.file` (e.g. `pkg/mod.py`)"
        );
    }
    let is_py = rel_file.ends_with(".py");
    let (clean_new, stripped_new) = strip_gutter(&block.new_text);

    if let Some(elided) = has_elision_placeholder(&clean_new) {
        bail!(
            "refused patch for `{rel_file}` containing placeholder comment `{elided}`: never elide existing code with `# ...`; pass `old_string` and `new_string` to replace only the changed lines"
        );
    }

    // Mode 1: Create new file (`create=1` explicitly requested)
    if block.create {
        let mut body = clean_new;
        if is_py && check_python_syntax(&body).is_some() {
            let unesc = unescape_once(&body);
            if check_python_syntax(&unesc).is_none() {
                body = unesc;
            } else if let Some((ln, snip)) = check_python_syntax(&body) {
                bail!("SyntaxError in new file `{rel_file}` near line {ln} (`{snip}`)");
            }
        }
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        if let Some(p) = full_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        crate::write_lf(&full_path, &body)?;
        record_created_file(env, rel_file);
        return Ok((
            rel_file.to_string(),
            format!("created `{rel_file}` ({} lines)", body.lines().count()),
        ));
    }

    ensure_base_snapshot(env, rel_file);
    let live_src = crate::read_lf(&full_path)?;
    let base_path = crate::join_rel(&env.agent_dir.join("base"), rel_file);
    let base_src = crate::read_lf(&base_path).unwrap_or_else(|_| live_src.clone());

    // Mode 2: Append to file (`append=1`)
    if block.append {
        let mut combined = live_src.clone();
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&clean_new);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
        if is_py && let Some((ln, snip)) = check_python_syntax(&combined) {
            bail!("SyntaxError after append in `{rel_file}` at line {ln} (`{snip}`)");
        }
        crate::write_lf(&full_path, &combined)?;
        return Ok((rel_file.to_string(), format!("appended to `{rel_file}`")));
    }

    // Mode 3: Add missing imports (`imports="..."`)
    if let Some(imp_text) = &block.imports {
        let updated = insert_imports(&live_src, imp_text);
        if is_py && let Some((ln, snip)) = check_python_syntax(&updated) {
            bail!("SyntaxError after imports in `{rel_file}` at line {ln} (`{snip}`)");
        }
        crate::write_lf(&full_path, &updated)?;
        return Ok((
            rel_file.to_string(),
            format!("added imports to `{rel_file}`"),
        ));
    }

    if clean_new.trim().is_empty() && !block.delete {
        bail!(
            "refused empty patch for `{rel_file}`: no `new_string` or `body` received inside `args` — ALL patch parameters (`\"file\"`, `\"symbol\"`, `\"old_string\"`, `\"new_string\"`, `\"body\"`) MUST be nested INSIDE `\"args\": {{...}}` (top-level keys outside `\"args\"` are ignored; pass `\"delete\": \"1\"` inside `\"args\"` only to delete)"
        );
    }
    if let Some(marker) = has_stray_patch_marker(&clean_new) {
        bail!(
            "refused malformed patch for `{rel_file}` containing stray marker `{marker}`: pass `old_string` and `new_string` cleanly in `args`"
        );
    }

    // Mode 4A (tried first): snippet replacement (`old_string` / `old` / `search` -> `new_string` / `body`),
    // optionally scoped to `symbol="Class.method"` or `start_line=N` so that a snippet occurring
    // several times is unambiguous
    if let Some(raw_old) = block.old.as_deref().filter(|s| !s.is_empty()) {
        let (clean_old, _) = strip_gutter(raw_old);
        if clean_old == clean_new {
            bail!(
                "no changes to apply in `{rel_file}`: `old_string` and `new_string` are identical"
            );
        }
        let defs = if is_py {
            parse_python_file(rel_file, &live_src).0
        } else {
            Vec::new()
        };
        let scoped_range = block
            .symbol
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .and_then(|sym| find_best_symbol(&defs, sym))
            .map(|d| (d.start_line, d.end_line));

        // 1. Try matching inside `symbol` range first (if `symbol` was given), then whole file
        let matches = find_snippet_matches_with_unescape(&live_src, &clean_old, scoped_range);
        let matches = if matches.is_empty() && scoped_range.is_some() {
            find_snippet_matches_with_unescape(&live_src, &clean_old, None)
        } else {
            matches
        };
        // 2. One unique window that differs only by 1-2 small slips (`*` for `*,`): adopt the file's
        // lines in `old_string` and the same corrections in `new_string`
        let (clean_old, clean_new, fuzzy_note, matches) = match matches.is_empty() {
            true => match fuzzy_realign(&live_src, &clean_old, &clean_new, scoped_range) {
                Some((fo, fnew, note)) => {
                    let m = find_all_snippet_matches(&live_src, &fo, None);
                    (fo, fnew, note, m)
                }
                None => match blankless_realign(&live_src, &clean_old) {
                    Some(window) => {
                        let m = find_all_snippet_matches(&live_src, &window, None);
                        (
                            window,
                            clean_new.clone(),
                            "old_string matched ignoring blank lines".to_string(),
                            m,
                        )
                    }
                    None => (clean_old, clean_new.clone(), String::new(), matches),
                },
            },
            false => (clean_old, clean_new.clone(), String::new(), matches),
        };

        if matches.is_empty() {
            // Omitted-docstring bridge: if `old_string` copied signature + body across an omitted docstring
            if is_py
                && let Some((bridged_src, bridge_note)) = try_splice_across_omitted_docstring(
                    rel_file, &live_src, &defs, &clean_old, &clean_new,
                )?
            {
                ensure_semantic_safety(
                    rel_file,
                    &live_src,
                    &bridged_src,
                    issue_text,
                    block.delete || block.force,
                )?;
                crate::write_lf(&full_path, &bridged_src)?;
                return Ok((rel_file.to_string(), bridge_note));
            }
            // old_string matches the ORIGINAL file but an earlier edit changed that region: apply only the
            // changed lines onto the current file when they occur exactly once there
            if find_all_snippet_matches(&base_src, &clean_old, None).len() == 1
                && let Some(rebased) = rebase_stale_edit(&live_src, &clean_old, &clean_new)
            {
                if is_py && let Some((ln, snip)) = check_python_syntax(&rebased) {
                    bail!(
                        "SyntaxError after applying your edit to the current file at line {ln} (`{snip}`): your old_string is from the original file and an earlier edit of yours changed it; view the current code first"
                    );
                }
                ensure_semantic_safety(
                    rel_file,
                    &live_src,
                    &rebased,
                    issue_text,
                    block.delete || block.force,
                )?;
                crate::write_lf(&full_path, &rebased)?;
                return Ok((
                    rel_file.to_string(),
                    "applied onto the current file: your old_string was the ORIGINAL code, an earlier edit of yours had changed it; only your changed lines were applied".to_string(),
                ));
            }
            // old text equal to one live window apart from joined string literals, spacing and doubled
            // backslashes: apply there
            if let Some(loose) = loose_realign(&live_src, &clean_old, &clean_new) {
                if is_py && let Some((ln, snip)) = check_python_syntax(&loose) {
                    bail!(
                        "SyntaxError after applying your edit at line {ln} (`{snip}`): your old_string differed from the file in spacing, string line breaks or backslashes; view the current code and copy it exactly"
                    );
                }
                ensure_semantic_safety(
                    rel_file,
                    &live_src,
                    &loose,
                    issue_text,
                    block.delete || block.force,
                )?;
                crate::write_lf(&full_path, &loose)?;
                return Ok((
                    rel_file.to_string(),
                    "old_string matched after normalising spacing, string line breaks and doubled backslashes (`\\\\` in your text, `\\` in the file); the file's lines were replaced".to_string(),
                ));
            }
            // Same docstring in old and new, but not the file's (abbreviated): edit around it, keep the file's
            if is_py
                && let Some(split_src) = edit_around_docstring(&live_src, &clean_old, &clean_new)
            {
                ensure_semantic_safety(
                    rel_file,
                    &live_src,
                    &split_src,
                    issue_text,
                    block.delete || block.force,
                )?;
                crate::write_lf(&full_path, &split_src)?;
                return Ok((
                    rel_file.to_string(),
                    "edited around the docstring: your old/new docstring differed from the file's, the file's docstring was kept".to_string(),
                ));
            }
            // the same edit resent after it applied: old_string is gone because new_string is
            // in place
            if already_applied(&live_src, &clean_new)
                || already_applied_line(&live_src, &clean_old, &clean_new)
            {
                return Ok((
                    rel_file.to_string(),
                    format!(
                        "ALREADY APPLIED: `new_string` is already in `{rel_file}` (this edit was made earlier); do NOT resend it. Next: `scripts/verify.py`"
                    ),
                ));
            }
            // show the part of the symbol the edit was aiming at, not its first 25 lines;
            // with no (or a wrong) `symbol`, search the whole file
            let sym_def = block
                .symbol
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .and_then(|sym| find_best_symbol(&defs, sym));
            let n_lines = live_src.lines().count().max(1);
            let (lo, hi, label) = match sym_def {
                Some(d) => (d.start_line, d.end_line, format!(" (`{}`)", d.qualname)),
                None => (1, n_lines, String::new()),
            };
            let (p_start, p_end) = closest_window(&live_src, &clean_old, lo, hi, 25);
            let live_slice = crate::format_numbered_slice(&live_src, p_start, p_end);
            // the same old_string that still does not match: offer the line range instead
            let by_lines = (hi - lo <= 3000)
                .then(|| best_line_window(&live_src, &clean_old, lo, hi))
                .flatten()
                .map(|(a, b)| format!(
                    "\nOr replace by line numbers (no `old_string`, no `symbol`): `args={{\"file\": \"{rel_file}\", \"start_line\": {a}, \"end_line\": {b}, \"new_string\": \"...\"}}` replaces lines {a}..{b} with your new code."
                ))
                .unwrap_or_default();
            bail!(
                "`old_string` did not match in `{rel_file}`{label}. Closest live lines {p_start}..{p_end}:\n{live_slice}Copy 2-8 exact unchanged lines from above into `old_string`!{by_lines}"
            );
        }

        let chosen = if matches.len() == 1 {
            matches[0]
        } else if let Some(hint_line) = block.start {
            *matches
                .iter()
                .min_by_key(|m| m.3.abs_diff(hint_line))
                .expect("non-empty")
        } else {
            let locs: Vec<String> = matches
                .iter()
                .enumerate()
                .take(5)
                .map(|(idx, &(_, _, _, s_ln, e_ln))| {
                    let sym_lbl = defs
                        .iter()
                        .filter(|d| d.start_line <= s_ln && e_ln <= d.end_line)
                        .max_by_key(|d| d.start_line)
                        .map(|d| format!(" in `{}`", d.qualname))
                        .unwrap_or_default();
                    format!("({}) lines {s_ln}-{e_ln}{sym_lbl}", idx + 1)
                })
                .collect();
            bail!(
                "`old_string` occurs {} times in `{rel_file}`: {}. Re-call `scripts/patch.py` adding `\"symbol\": \"<Class.method>\"` (or `\"start_line\": <line>`) in `args` to disambiguate!",
                matches.len(),
                locs.join(", ")
            );
        };

        let (span_start, span_end, matched_indent, s_ln, e_ln) = chosen;
        let at_line_start =
            span_start == 0 || live_src.as_bytes().get(span_start - 1) == Some(&b'\n');
        let min_old_indent = clean_old
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(line_indent)
            .min()
            .unwrap_or(0);
        let min_new_indent = clean_new
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(line_indent)
            .min()
            .unwrap_or(0);
        let prepared_new = if at_line_start {
            if min_new_indent == 0 && matched_indent > 0 && !block.delete {
                shift_indent(&clean_new, matched_indent)
            } else {
                clean_new.clone()
            }
        } else if min_old_indent == 0 && min_new_indent > 0 && !clean_new.contains('\n') {
            clean_new.trim_start_matches(' ').to_string()
        } else {
            clean_new.clone()
        };

        // Guard against partial compound-block overwrite (e.g. `old_string` matched only the first 1-2 lines of
        // a multi-line `if prefix:` block while indented body lines remain in `&live_src[span_end..]`, and
        // `new_string` drops `if prefix:` and ends with `raise`/`return`, silently turning the remaining block
        // into unreachable dead code).
        if is_py
            && at_line_start
            && !block.delete
            && !block.force
            && let Some(first_old_trim) = clean_old
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#'))
            && is_compound_suite_header(first_old_trim)
            && let Some(next_after_ln) = live_src[span_end..]
                .lines()
                .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            && line_indent(next_after_ln) > matched_indent
            && !prepared_new.lines().any(|l| l.trim() == first_old_trim)
            && let Some(last_new_trim) = prepared_new
                .lines()
                .rev()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#'))
            && is_control_flow_terminator(last_new_trim)
        {
            let next_trim = next_after_ln.trim();
            bail!(
                "refused partial block overwrite in `{rel_file}` (lines {s_ln}-{e_ln}): `old_string` matched only the top of `{first_old_trim}` while indented body line `{next_trim}` remains below line {e_ln}, so replacing `{first_old_trim}` with `{last_new_trim}` makes the remaining block unreachable dead code! To insert a guard before `{first_old_trim}`, keep the unchanged `old_string` lines at the end of `new_string`!"
            );
        }

        let preserve_trailing_nl = live_src[span_start..span_end].ends_with('\n');
        let assemble = |candidate: &str| -> String {
            let mut c = candidate.to_string();
            if preserve_trailing_nl && !c.is_empty() && !c.ends_with('\n') {
                c.push('\n');
            } else if !preserve_trailing_nl && c.ends_with('\n') && !clean_old.ends_with('\n') {
                c.pop();
                if c.ends_with('\r') {
                    c.pop();
                }
            }
            format!("{}{}{}", &live_src[..span_start], c, &live_src[span_end..])
        };
        let initial = assemble(&prepared_new);
        let repair_target_indent = if at_line_start {
            Some(matched_indent)
        } else {
            None
        };
        let (final_src, note) = if is_py {
            auto_repair_and_validate(initial, &prepared_new, repair_target_indent, assemble)
                .map_err(|e| {
                    let dangling_hint = if has_unclosed_open_delim(&clean_old) {
                        format!(
                            "\nHINT: `old_string` matched only lines {s_ln}-{e_ln} and ended with an unclosed `(`/`[`/`{{`, leaving the remaining lines of the multi-line signature or expression dangling below line {e_ln}! Include ALL lines of the multi-line signature through `):` in `old_string`, OR replace the whole method via `symbol=\"Class.method\"` + `body`."
                        )
                    } else {
                        String::new()
                    };
                    anyhow::anyhow!("{rel_file} (lines {s_ln}-{e_ln}): {e}{dangling_hint}")
                })?
        } else {
            (initial, String::new())
        };
        if is_py {
            ensure_semantic_safety(
                rel_file,
                &live_src,
                &final_src,
                issue_text,
                block.delete || block.force,
            )?;
        }
        crate::write_lf(&full_path, &final_src)?;
        let note = [note.as_str(), fuzzy_note.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join("; ");
        let note_tag = if note.is_empty() && !stripped_new {
            String::new()
        } else {
            format!(" ({note})")
        };
        return Ok((
            rel_file.to_string(),
            format!("updated lines {s_ln}-{e_ln} in `{rel_file}`{note_tag}"),
        ));
    }

    // Mode 5 (before Mode 4): insert after an AST symbol or module assignment (`after_symbol=Class.method` / `after_symbol=fn` / `after_symbol=CONST`),
    // even when `symbol="new_fn"` was also passed alongside `after_symbol`
    if let Some(after_sym) = block
        .after_symbol
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        let (defs, _) = parse_python_file(rel_file, &live_src);
        let dedented_new = dedent_block(&clean_new);
        let unesc_dedented = unescape_once(&dedented_new);
        let parse_src = if is_py
            && check_python_syntax(&dedented_new).is_some()
            && check_python_syntax(&unesc_dedented).is_none()
        {
            &unesc_dedented
        } else {
            &dedented_new
        };
        let top_new_defs: Vec<crate::ast_graph::DefNode> = if is_py {
            let (nd, _) = parse_python_file("<patch>", parse_src);
            nd.iter()
                .filter(|d| {
                    !nd.iter().any(|other| {
                        other.id != d.id
                            && d.start_line > other.start_line
                            && d.end_line <= other.end_line
                    })
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Idempotency guard: if `clean_new` defines a single function/method/class that already exists in `defs`
        // in the same scope, delegate to Mode 4 (`symbol=existing.qualname`) so re-patching updates it in-place
        // instead of inserting a duplicate definition
        let anchor_def_opt = find_best_symbol(&defs, after_sym);
        if is_py && top_new_defs.len() == 1 {
            let new_name = &top_new_defs[0].name;
            let expected_qual = if let Some(anc) = anchor_def_opt
                && let Some((cls_prefix, _)) = anc.qualname.rsplit_once('.')
            {
                format!("{cls_prefix}.{new_name}")
            } else {
                new_name.clone()
            };
            if let Some(existing) = defs.iter().find(|d| {
                d.qualname == expected_qual
                    || (!expected_qual.contains('.') && d.qualname == *new_name)
            }) {
                let mut delegated = block.clone();
                delegated.after_symbol = None;
                delegated.symbol = Some(existing.qualname.clone());
                return apply_single_block(env, &delegated, issue_text);
            }
        }

        let (anchor_label, anchor_start, anchor_end) = if let Some(d) = anchor_def_opt {
            (d.qualname.clone(), d.start_line, d.end_line)
        } else if is_py
            && let Some((lbl, s_ln, e_ln)) = find_module_assignment_span(&live_src, after_sym)
        {
            (lbl, s_ln, e_ln)
        } else {
            let avail: Vec<String> = defs.iter().take(15).map(|d| d.qualname.clone()).collect();
            bail!(
                "after_symbol `{after_sym}` not found in `{rel_file}` (available: {})",
                if avail.is_empty() {
                    "<no functions/classes in file>".to_string()
                } else {
                    avail.join(", ")
                }
            );
        };

        let lines: Vec<&str> = live_src.split_inclusive('\n').collect();
        let cut = anchor_end.min(lines.len());
        let target_indent = lines
            .get(anchor_start.saturating_sub(1))
            .map(|l| line_indent(l));
        let aligned_new = align_block_to_indent(&clean_new, target_indent);
        let assemble = |candidate: &str| -> String {
            let mut c = candidate.to_string();
            if !c.is_empty() && !c.ends_with('\n') {
                c.push('\n');
            }
            format!("{}\n{}{}", lines[..cut].concat(), c, lines[cut..].concat())
        };
        let initial = assemble(&aligned_new);
        let (final_src, _) = if is_py {
            auto_repair_and_validate(initial, &aligned_new, target_indent, assemble)
                .map_err(|e| anyhow::anyhow!("{rel_file} (after_symbol={after_sym}): {e}"))?
        } else {
            (initial, String::new())
        };
        if is_py {
            ensure_semantic_safety(
                rel_file,
                &live_src,
                &final_src,
                issue_text,
                block.delete || block.force,
            )?;
        }
        crate::write_lf(&full_path, &final_src)?;
        return Ok((
            rel_file.to_string(),
            format!("inserted after `{anchor_label}` in `{rel_file}`"),
        ));
    }

    // `symbol=` plus `start=`/`end=` and a body that is not a whole definition is a line-range edit inside the
    // symbol: leave it to Mode 8
    let range_edit = block.start.is_some()
        && block.end.is_some()
        && !clean_new
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(str::trim_start)
            .is_some_and(|l| {
                l.starts_with("def ") || l.starts_with("async def ") || l.starts_with("class ")
            });
    // Mode 4: AST symbol replacement (`symbol=Class.method` or `symbol=func`) with upsert and mass-deletion guards
    if let Some(sym) = block
        .symbol
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .filter(|_| !range_edit)
    {
        let (defs, _) = parse_python_file(rel_file, &live_src);
        let dedented_new = dedent_block(&clean_new);
        let unesc_dedented = unescape_once(&dedented_new);
        let parse_src = if check_python_syntax(&dedented_new).is_some()
            && check_python_syntax(&unesc_dedented).is_none()
        {
            &unesc_dedented
        } else {
            &dedented_new
        };
        let (new_defs, _) = parse_python_file("<patch>", parse_src);
        let top_new_defs: Vec<&crate::ast_graph::DefNode> = new_defs
            .iter()
            .filter(|d| {
                !new_defs.iter().any(|other| {
                    other.id != d.id
                        && d.start_line > other.start_line
                        && d.end_line <= other.end_line
                })
            })
            .collect();

        // Check if `sym` is not found directly in `defs`
        let Some(mut target_def) = find_best_symbol(&defs, sym) else {
            // Case A: `symbol=ClassName.new_method` where `ClassName` exists in `defs` and `new_method` is a new method
            if let Some((cls_part, meth_part)) = sym.trim().rsplit_once('.')
                && let Some(cls_def) = find_best_symbol(&defs, cls_part)
                && cls_def.kind == "class"
                && top_new_defs.len() == 1
                && top_new_defs[0].kind != "class"
                && top_new_defs[0].name == meth_part
            {
                return insert_method_into_class(
                    &full_path, rel_file, &live_src, cls_def, &clean_new, meth_part,
                );
            }

            // Case C: additive top-level function/class upsert into an existing module.
            // Handles both `symbol="new_fn"` + `body="def new_fn(...)..."` and
            // `symbol="pkg/types.py"` / `symbol="types"` + `body="def helper(...)..."`
            // without overwriting existing module code or constants
            let mod_stem = Path::new(rel_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let sym_matches_new_def = !sym.contains('.')
                && top_new_defs
                    .iter()
                    .any(|d| d.name.eq_ignore_ascii_case(sym.trim()));
            let sym_matches_module =
                sym.trim().eq_ignore_ascii_case(mod_stem) || crate::clean_rel_arg(sym) == rel_file;
            let no_name_collision = top_new_defs
                .iter()
                .all(|nd| !defs.iter().any(|ed| ed.qualname == nd.name));
            if is_py
                && !top_new_defs.is_empty()
                && check_python_syntax(parse_src).is_none()
                && (sym_matches_new_def || sym_matches_module)
                && no_name_collision
                && !looks_like_unbound_method(parse_src, &top_new_defs)
            {
                let inserted_src = insert_top_level_block(&live_src, parse_src);
                if check_python_syntax(&inserted_src).is_none()
                    && ensure_semantic_safety(
                        rel_file,
                        &live_src,
                        &inserted_src,
                        issue_text,
                        block.delete || block.force,
                    )
                    .is_ok()
                {
                    crate::write_lf(&full_path, &inserted_src)?;
                    let names: Vec<&str> = top_new_defs.iter().map(|d| d.name.as_str()).collect();
                    return Ok((
                        rel_file.to_string(),
                        format!(
                            "inserted top-level symbol `{}` into `{rel_file}`",
                            names.join(", ")
                        ),
                    ));
                }
            }

            // Case B: `symbol=<module_stem>` on a small constant/type-only module,
            // or any new top-level class/function added to an empty/docstring-only stub module (`defs.is_empty && live_line_count <= 20`)
            let live_line_count = live_src.lines().count();
            let is_stub_upsert =
                defs.is_empty() && live_line_count <= 20 && !top_new_defs.is_empty();
            if ((sym.trim().eq_ignore_ascii_case(mod_stem)
                || crate::clean_rel_arg(sym) == rel_file)
                && defs.is_empty()
                && live_line_count <= 60)
                || is_stub_upsert
            {
                let trimmed_new = clean_new.trim();
                // Single `from X import ...` update against matching `from X import ...` line in `live_src`
                if trimmed_new.lines().count() == 1 && trimmed_new.starts_with("from ") {
                    let prefix = trimmed_new
                        .split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" ");
                    let matches: Vec<&str> = live_src
                        .lines()
                        .filter(|l| l.trim_start().starts_with(&prefix))
                        .collect();
                    if matches.len() == 1 {
                        let old_line = matches[0];
                        let updated = live_src.replacen(old_line, trimmed_new, 1);
                        if check_python_syntax(&updated).is_none() {
                            crate::write_lf(&full_path, &updated)?;
                            return Ok((
                                rel_file.to_string(),
                                format!("updated import `{prefix} ...` in `{rel_file}`"),
                            ));
                        }
                    }
                }
                if clean_new.lines().count() * 2 >= live_line_count {
                    let mut body = clean_new.clone();
                    if check_python_syntax(&body).is_some() {
                        let u = unescape_once(&body);
                        if check_python_syntax(&u).is_none() {
                            body = u;
                        }
                    }
                    if !body.ends_with('\n') {
                        body.push('\n');
                    }
                    if check_python_syntax(&body).is_none() {
                        crate::write_lf(&full_path, &body)?;
                        return Ok((rel_file.to_string(), format!("updated module `{rel_file}`")));
                    }
                }
            }

            let avail: Vec<String> = defs.iter().take(15).map(|d| d.qualname.clone()).collect();
            bail!(
                "symbol `{sym}` not found in `{rel_file}` (available: {})",
                if avail.is_empty() {
                    "<no functions/classes in file — pass `old_string` and `new_string`>"
                        .to_string()
                } else {
                    avail.join(", ")
                }
            );
        };

        // Guard 4A: `target_def` is a `class`, but `clean_new` does not define `class <target_def.name>`
        if !block.delete
            && target_def.kind == "class"
            && !top_new_defs
                .iter()
                .any(|d| d.kind == "class" && d.name == target_def.name)
        {
            if !top_new_defs.is_empty() && top_new_defs.iter().all(|d| d.kind != "class") {
                if top_new_defs.len() == 1 {
                    let m_name = &top_new_defs[0].name;
                    if let Some(existing_method) = defs.iter().find(|d| {
                        d.start_line > target_def.start_line
                            && d.end_line <= target_def.end_line
                            && (d.name == *m_name
                                || d.qualname == format!("{}.{}", target_def.qualname, m_name))
                    }) {
                        target_def = existing_method;
                    } else {
                        return insert_method_into_class(
                            &full_path, rel_file, &live_src, target_def, &clean_new, m_name,
                        );
                    }
                } else {
                    // Multiple methods provided for `target_def` class: upsert each method individually
                    let src_lines: Vec<&str> = parse_src.split_inclusive('\n').collect();
                    let cls_qualname = target_def.qualname.clone();
                    let mut sub_msgs = Vec::new();
                    for md in &top_new_defs {
                        let s = md.start_line.saturating_sub(1).min(src_lines.len());
                        let e = md.end_line.min(src_lines.len());
                        let m_slice = src_lines[s..e].concat();
                        let sub_block = PatchBlock {
                            file: rel_file.to_string(),
                            symbol: Some(format!("{cls_qualname}.{}", md.name)),
                            new_text: m_slice,
                            force: block.force,
                            ..Default::default()
                        };
                        let (_, msg) = apply_single_block(env, &sub_block, issue_text)?;
                        sub_msgs.push(msg);
                    }
                    return Ok((rel_file.to_string(), sub_msgs.join("; ")));
                }
            } else {
                let span_len = target_def.end_line.saturating_sub(target_def.start_line) + 1;
                bail!(
                    "refused to overwrite entire class `{}` ({span_len} lines) in `{rel_file}` because `body` does not define `class {}`. Pass `\"symbol\": \"{}.<method_name>\"` (with `def <method_name>(self, ...):`) or pass `\"old_string\"` + `\"new_string\"`",
                    target_def.qualname,
                    target_def.name,
                    target_def.qualname
                );
            }
        }

        // Guard 4B: `target_def` is a function/method, check that `clean_new` defines `target_def.name` (or redirect to nested/sibling function)
        if !block.delete
            && is_py
            && target_def.kind != "class"
            && !new_defs.iter().any(|d| d.name == target_def.name)
        {
            if top_new_defs.len() == 1 && top_new_defs[0].kind != "class" {
                let m_name = &top_new_defs[0].name;
                // 1. Check for a nested helper function inside `target_def`
                if let Some(nested) = defs.iter().find(|d| {
                    d.start_line > target_def.start_line
                        && d.end_line <= target_def.end_line
                        && d.name == *m_name
                }) {
                    target_def = nested;
                } else if let Some((cls_prefix, _)) = target_def.qualname.rsplit_once('.')
                    && let Some(sibling) = defs
                        .iter()
                        .find(|d| d.qualname == format!("{cls_prefix}.{m_name}"))
                {
                    target_def = sibling;
                } else {
                    bail!(
                        "refused to overwrite `{}` in `{rel_file}` with `def {m_name}` (different function name). Pass `\"symbol\": \"{m_name}\"` or `\"after_symbol\": \"{}\"` (to insert a new function without deleting `{}`)!",
                        target_def.qualname,
                        target_def.qualname,
                        target_def.qualname
                    );
                }
            } else {
                let span_len = target_def.end_line.saturating_sub(target_def.start_line) + 1;
                bail!(
                    "refused to overwrite function `{}` ({span_len} lines) in `{rel_file}` because `body` omitted `def {}(...):`. To edit specific lines inside `{}`, pass `\"symbol\": \"{}\"`, `\"old_string\": \"<exact lines>\"`, and `\"new_string\": \"<replacement lines>\"`!",
                    target_def.qualname,
                    target_def.name,
                    target_def.qualname,
                    target_def.qualname
                );
            }
        }

        // Guard 4C: anti-truncation / mass-deletion guard on `target_def`
        let old_span_lines = target_def.end_line.saturating_sub(target_def.start_line) + 1;
        let new_nonempty_lines = clean_new.lines().filter(|l| !l.trim().is_empty()).count();
        if !block.delete
            && !block.force
            && old_span_lines >= 35
            && new_nonempty_lines * 3 < old_span_lines
            && old_span_lines.saturating_sub(new_nonempty_lines) >= 30
        {
            bail!(
                "refused destructive truncation of `{}` in `{rel_file}` ({old_span_lines} lines -> {new_nonempty_lines} lines). Pass `\"symbol\": \"{}\"`, `\"old_string\": \"<exact lines>\"`, and `\"new_string\": \"<replacement lines>\"` to edit only the changed lines!",
                target_def.qualname,
                target_def.qualname
            );
        }

        let lines: Vec<&str> = live_src.split_inclusive('\n').collect();
        let at = target_def.start_line.saturating_sub(1);
        let cut = target_def.end_line.min(lines.len());
        let target_indent = lines.get(at).map(|l| line_indent(l));
        let aligned_new = align_block_to_indent(&clean_new, target_indent);

        let assemble = |candidate: &str| -> String {
            let mut c = candidate.to_string();
            if !c.is_empty() && !c.ends_with('\n') {
                c.push('\n');
            }
            format!("{}{}{}", lines[..at].concat(), c, lines[cut..].concat())
        };
        let initial = assemble(&aligned_new);
        let (final_src, mut note) = if is_py {
            auto_repair_and_validate(initial, &aligned_new, target_indent, assemble)
                .map_err(|e| anyhow::anyhow!("{rel_file} (symbol={sym}): {e}"))?
        } else {
            (initial, String::new())
        };
        if aligned_new != clean_new && note.is_empty() {
            note = format!("aligned to indent {}", target_indent.unwrap_or(0));
        }

        // Guard 4D: class-structure, qualname and semantic safety post-condition
        if is_py {
            let (after_defs, _) = parse_python_file(rel_file, &final_src);
            if !block.delete && !after_defs.iter().any(|d| d.qualname == target_def.qualname) {
                bail!(
                    "indentation/scope mismatch after replacing `{}` in `{rel_file}`: `{}` lost its enclosing scope; file kept unchanged",
                    target_def.qualname,
                    target_def.qualname
                );
            }
            if let Some((parent_qual, _)) = target_def.qualname.rsplit_once('.') {
                let prefix = format!("{parent_qual}.");
                let before_members = defs
                    .iter()
                    .filter(|d| d.qualname.starts_with(&prefix))
                    .count();
                let after_members = after_defs
                    .iter()
                    .filter(|d| d.qualname.starts_with(&prefix))
                    .count();
                let min_expected = if block.delete {
                    before_members.saturating_sub(1)
                } else {
                    before_members
                };
                if after_members < min_expected {
                    bail!(
                        "indentation/scope mismatch in class `{parent_qual}` (`{rel_file}`): member count dropped from {before_members} to {after_members}; file kept unchanged"
                    );
                }
            }
            ensure_semantic_safety(
                rel_file,
                &live_src,
                &final_src,
                issue_text,
                block.delete || block.force,
            )?;
        }

        crate::write_lf(&full_path, &final_src)?;
        let note_tag = if note.is_empty() {
            String::new()
        } else {
            format!(" ({note})")
        };
        return Ok((
            rel_file.to_string(),
            format!(
                "replaced symbol `{}` in `{rel_file}`:{}-{}{}",
                target_def.qualname, target_def.start_line, target_def.end_line, note_tag
            ),
        ));
    }

    // Mode 7: Insert after line `after=N`
    if let Some(after_line) = block.after {
        let lines: Vec<&str> = live_src.split_inclusive('\n').collect();
        let cut = after_line.min(lines.len());
        let target_indent = lines.get(cut.saturating_sub(1)).map(|l| line_indent(l));
        let assemble = |candidate: &str| -> String {
            let mut c = candidate.to_string();
            if !c.is_empty() && !c.ends_with('\n') {
                c.push('\n');
            }
            format!("{}{}{}", lines[..cut].concat(), c, lines[cut..].concat())
        };
        let initial = assemble(&clean_new);
        let (final_src, _) = if is_py {
            auto_repair_and_validate(initial, &clean_new, target_indent, assemble)
                .map_err(|e| anyhow::anyhow!("{rel_file} (after={after_line}): {e}"))?
        } else {
            (initial, String::new())
        };
        crate::write_lf(&full_path, &final_src)?;
        return Ok((
            rel_file.to_string(),
            format!("inserted after line {after_line} in `{rel_file}`"),
        ));
    }

    // Mode 8: line-range replacement (`start=N end=M`) with 3-way merge when the file has already changed
    if let (Some(start), Some(end)) = (block.start, block.end.or(block.start)) {
        if start == 0 || end < start {
            bail!("invalid line range start={start} end={end} in `{rel_file}`");
        }
        // Check docstring guard
        if is_py
            && let Some((s_str, e_str, fn_body)) = inside_multiline_string(&live_src, start, end)
        {
            let hint = fn_body
                .map(|b| format!(" Enclosing function code starts at line {b}."))
                .unwrap_or_default();
            bail!(
                "lines {start}-{end} in `{rel_file}` are inside a docstring/string literal ({s_str}-{e_str}).{hint}"
            );
        }

        let live_lines: Vec<&str> = live_src.split_inclusive('\n').collect();
        let base_lines: Vec<&str> = base_src.split_inclusive('\n').collect();
        let at_base = start.saturating_sub(1).min(base_lines.len());
        let cut_base = end.min(base_lines.len());
        let at_live = start.saturating_sub(1).min(live_lines.len());
        let cut_live = end.min(live_lines.len());

        let base_slice = base_lines[at_base..cut_base].concat();
        let live_slice = live_lines[at_live..cut_live].concat();

        if live_src != base_src && base_slice != live_slice && !base_slice.is_empty() {
            // 1. Try 3-way `git merge-file` via staged edit on `base_src`
            let target_indent = base_lines.get(at_base).map(|l| line_indent(l));
            let assemble_base = |candidate: &str| -> String {
                let mut c = candidate.to_string();
                if !c.is_empty() && !c.ends_with('\n') {
                    c.push('\n');
                }
                format!(
                    "{}{}{}",
                    base_lines[..at_base].concat(),
                    c,
                    base_lines[cut_base..].concat()
                )
            };
            let staged_init = assemble_base(&clean_new);
            if let Ok((staged_valid, _)) =
                auto_repair_and_validate(staged_init, &clean_new, target_indent, assemble_base)
                && let Some(merged) =
                    three_way_merge_strings(env, rel_file, &live_src, &base_src, &staged_valid)
            {
                if is_py {
                    ensure_semantic_safety(
                        rel_file,
                        &live_src,
                        &merged,
                        issue_text,
                        block.delete || block.force,
                    )?;
                }
                crate::write_lf(&full_path, &merged)?;
                return Ok((
                    rel_file.to_string(),
                    format!(
                        "3-way merged lines {start}-{end} into `{rel_file}` (preserving prior edits)"
                    ),
                ));
            }
            // 2. Fallback: locate `base_slice` inside `live_src` at its shifted position
            if let Some(pos) = live_src.find(&base_slice) {
                let span_end = pos + base_slice.len();
                let assemble_shifted = |candidate: &str| -> String {
                    let mut c = candidate.to_string();
                    if !c.is_empty() && !c.ends_with('\n') {
                        c.push('\n');
                    }
                    format!("{}{}{}", &live_src[..pos], c, &live_src[span_end..])
                };
                let init_s = assemble_shifted(&clean_new);
                if let Ok((valid_s, _)) =
                    auto_repair_and_validate(init_s, &clean_new, target_indent, assemble_shifted)
                {
                    if is_py {
                        ensure_semantic_safety(
                            rel_file,
                            &live_src,
                            &valid_s,
                            issue_text,
                            block.delete || block.force,
                        )?;
                    }
                    crate::write_lf(&full_path, &valid_s)?;
                    return Ok((
                        rel_file.to_string(),
                        format!(
                            "applied shifted range {start}-{end} in `{rel_file}` (preserving prior edits)"
                        ),
                    ));
                }
            }
        }

        let target_indent = live_lines.get(at_live).map(|l| line_indent(l));
        let assemble_live = |candidate: &str| -> String {
            let mut c = candidate.to_string();
            if !c.is_empty() && !c.ends_with('\n') {
                c.push('\n');
            }
            format!(
                "{}{}{}",
                live_lines[..at_live].concat(),
                c,
                live_lines[cut_live..].concat()
            )
        };
        let initial = assemble_live(&clean_new);
        match auto_repair_and_validate(initial, &clean_new, target_indent, assemble_live) {
            Ok((final_src, note)) => {
                if is_py {
                    ensure_semantic_safety(
                        rel_file,
                        &live_src,
                        &final_src,
                        issue_text,
                        block.delete || block.force,
                    )?;
                }
                crate::write_lf(&full_path, &final_src)?;
                let note_tag = if note.is_empty() {
                    String::new()
                } else {
                    format!(" ({note})")
                };
                return Ok((
                    rel_file.to_string(),
                    format!("replaced lines {start}-{end} in `{rel_file}`{note_tag}"),
                ));
            }
            Err(err_msg) => {
                // `insert_instead` fallback
                let mut c = clean_new.clone();
                if !c.ends_with('\n') {
                    c.push('\n');
                }
                for (where_tag, cand) in [
                    (
                        "before",
                        format!(
                            "{}{}{}",
                            live_lines[..at_live].concat(),
                            c,
                            live_lines[at_live..].concat()
                        ),
                    ),
                    (
                        "after",
                        format!(
                            "{}{}{}",
                            live_lines[..cut_live].concat(),
                            c,
                            live_lines[cut_live..].concat()
                        ),
                    ),
                ] {
                    if check_python_syntax(&cand).is_none()
                        && (!is_py
                            || ensure_semantic_safety(
                                rel_file,
                                &live_src,
                                &cand,
                                issue_text,
                                block.delete || block.force,
                            )
                            .is_ok())
                    {
                        crate::write_lf(&full_path, &cand)?;
                        return Ok((
                            rel_file.to_string(),
                            format!(
                                "inserted {where_tag} lines {start}-{end} in `{rel_file}` (replacing broke syntax)"
                            ),
                        ));
                    }
                }
                bail!("{rel_file} (lines {start}-{end}): {err_msg}");
            }
        }
    }

    bail!(
        "patch block for `{rel_file}` must specify `old_string` + `new_string` (with optional `symbol`), `symbol` + `body`, `after_symbol` + `body`, `start_line` + `end_line` + `body`, `imports`, or `create=1`"
    )
}

/// Reject edits before writing to disk if they delete an existing public definition
/// or splice code inside an existing multi-line `"""..."""` docstring.
fn ensure_semantic_safety(
    rel_file: &str,
    before_src: &str,
    after_src: &str,
    issue_text: &str,
    allow_delete: bool,
) -> Result<()> {
    if allow_delete {
        return Ok(());
    }
    let deleted = check_deleted_public_defs(rel_file, before_src, after_src, issue_text);
    if let Some(first) = deleted.first() {
        bail!(
            "refused edit because it deletes an existing public definition ({first})! Keep existing functions/methods intact (to add a new method to a class, pass `\"symbol\": \"ClassName.new_method\"` with `\"body\": \"def new_method(self, ...): ...\"`; to add a new function, pass `\"after_symbol\": \"existing_fn\"`; or pass `\"force\": \"1\"` ONLY if deletion is intentional)"
        );
    }
    let spliced = check_docstring_code_splice(rel_file, before_src, after_src);
    if let Some(first) = spliced.first() {
        bail!("{first} (file kept unchanged)");
    }
    Ok(())
}

fn delim_delta_line(line: &str) -> isize {
    let mut delta = 0isize;
    let mut in_quote: Option<char> = None;
    let mut escaped = false;
    for ch in line.chars() {
        if let Some(q) = in_quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                in_quote = None;
            }
            continue;
        }
        match ch {
            '#' => break,
            '\'' | '"' => in_quote = Some(ch),
            '(' | '[' | '{' => delta += 1,
            ')' | ']' | '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

fn has_unclosed_open_delim(text: &str) -> bool {
    let mut bal = 0isize;
    for ln in text.lines() {
        bal += delim_delta_line(ln);
    }
    bal > 0
}

pub(crate) fn find_module_assignment_span(
    src: &str,
    raw_ident: &str,
) -> Option<(String, usize, usize)> {
    let ident = raw_ident.trim();
    if ident.is_empty() || !ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let lines: Vec<&str> = src.lines().collect();
    for (idx, ln) in lines.iter().enumerate() {
        let t = ln.trim_start();
        let Some(rest) = t.strip_prefix(ident) else {
            continue;
        };
        let rest_t = rest.trim_start();
        let is_assign = if let Some(after_eq) = rest_t.strip_prefix('=') {
            !after_eq.starts_with('=')
        } else if let Some(after_colon) = rest_t.strip_prefix(':') {
            after_colon.contains('=')
        } else {
            false
        };
        if !is_assign {
            continue;
        }
        let start_ln = idx + 1;
        let mut bal = 0isize;
        let mut end_ln = start_ln;
        for (j, cont_ln) in lines[idx..].iter().enumerate() {
            bal += delim_delta_line(cont_ln);
            end_ln = idx + j + 1;
            if bal <= 0 && !cont_ln.trim_end().ends_with('\\') {
                break;
            }
        }
        return Some((ident.to_string(), start_ln, end_ln));
    }
    None
}

fn looks_like_unbound_method(src: &str, top_defs: &[&crate::ast_graph::DefNode]) -> bool {
    let lines: Vec<&str> = src.lines().collect();
    for d in top_defs {
        if d.kind == "class" {
            continue;
        }
        let s = d.start_line.saturating_sub(1).min(lines.len());
        let e = (d.start_line + 3).min(d.end_line).min(lines.len());
        let sig = lines[s..e].join(" ");
        if let Some((_, params)) = sig.split_once('(') {
            let p_trim = params.trim_start();
            for kw in ["self", "cls"] {
                if let Some(after) = p_trim.strip_prefix(kw)
                    && after
                        .chars()
                        .next()
                        .is_some_and(|c| matches!(c, ',' | ')' | ':' | ' ' | '\t'))
                {
                    return true;
                }
            }
        }
    }
    false
}

fn insert_top_level_block(live_src: &str, block_src: &str) -> String {
    let mut clean_block = dedent_block(block_src);
    // Strip duplicate `from __future__ import annotations` if `live_src` already has code
    if !live_src.trim().is_empty() {
        let filtered: Vec<&str> = clean_block
            .lines()
            .filter(|l| l.trim() != "from __future__ import annotations")
            .collect();
        clean_block = filtered.join("\n");
    }
    let clean_block = clean_block.trim();
    if clean_block.is_empty() {
        return live_src.to_string();
    }
    let lines: Vec<&str> = live_src.split_inclusive('\n').collect();
    if let Some(main_idx) = lines
        .iter()
        .position(|l| l.starts_with("if __name__") && l.contains("__main__"))
    {
        let before = lines[..main_idx].concat();
        let after = lines[main_idx..].concat();
        return format!("{}\n\n{clean_block}\n\n{after}", before.trim_end());
    }
    let trimmed_live = live_src.trim_end();
    if trimmed_live.is_empty() {
        format!("{clean_block}\n")
    } else {
        format!("{trimmed_live}\n\n{clean_block}\n")
    }
}

/// Find all `(byte_start, byte_end, first_line_indent, start_line_1based, end_line_1based)` matches
/// of `old` inside `src` (optionally restricted to `line_range` `1`-based inclusive).
fn find_all_snippet_matches(
    src: &str,
    old: &str,
    line_range: Option<(usize, usize)>,
) -> Vec<(usize, usize, isize, usize, usize)> {
    let old_norm = old.replace("\r\n", "\n");
    let src_lines: Vec<&str> = src.split_inclusive('\n').collect();
    if src_lines.is_empty() || old_norm.trim().is_empty() {
        return Vec::new();
    }
    let mut line_offsets = Vec::with_capacity(src_lines.len() + 1);
    let mut acc = 0usize;
    for l in &src_lines {
        line_offsets.push(acc);
        acc += l.len();
    }
    line_offsets.push(acc);

    let (min_idx, max_excl) = match line_range {
        Some((s, e)) => (
            s.saturating_sub(1).min(src_lines.len()),
            e.min(src_lines.len()),
        ),
        None => (0, src_lines.len()),
    };
    if min_idx >= max_excl {
        return Vec::new();
    }
    let region_start_b = line_offsets[min_idx];
    let region_end_b = line_offsets[max_excl];
    let region = &src[region_start_b..region_end_b];

    let byte_to_line_1based = |byte_pos: usize| -> usize {
        match line_offsets.binary_search(&byte_pos) {
            Ok(i) => i + 1,
            Err(i) => i.max(1),
        }
    };

    let old_lines: Vec<&str> = old_norm.lines().collect();

    // 1A. Exact full-line match (including exact leading indentation, avoiding 4-space substring matching inside 8-space line)
    if !old_lines.is_empty() && old_lines.len() <= max_excl - min_idx {
        let mut exact_line_hits = Vec::new();
        for i in min_idx..=(max_excl - old_lines.len()) {
            if (0..old_lines.len()).all(|j| {
                src_lines[i + j].trim_end_matches(['\r', '\n'])
                    == old_lines[j].trim_end_matches('\r')
            }) {
                let start_b = line_offsets[i];
                let end_b = line_offsets[i + old_lines.len()];
                let indent = line_indent(src_lines[i]);
                exact_line_hits.push((start_b, end_b, indent, i + 1, i + old_lines.len()));
            }
        }
        if !exact_line_hits.is_empty() {
            return exact_line_hits;
        }
    }

    // 1B. Full-line whitespace-trimmed matches inside `min_idx..max_excl` (returns column-0 `start_b` so
    // unindented `old_string` + indented `new_string` never leaves leftover spaces before `span_start`)
    if !old_lines.is_empty()
        && old_lines.iter().any(|l| !l.trim().is_empty())
        && old_lines.len() <= max_excl - min_idx
    {
        let mut trimmed_hits = Vec::new();
        for i in min_idx..=(max_excl - old_lines.len()) {
            if (0..old_lines.len()).all(|j| src_lines[i + j].trim() == old_lines[j].trim()) {
                let start_b = line_offsets[i];
                let end_b = line_offsets[i + old_lines.len()];
                let indent = line_indent(src_lines[i]);
                trimmed_hits.push((start_b, end_b, indent, i + 1, i + old_lines.len()));
            }
        }
        if !trimmed_hits.is_empty() {
            return trimmed_hits;
        }
    }

    // 1C. Mid-line substring matches inside `region` (for inline expression edits on a single line, e.g. `Column(_index=index)`)
    for needle in [old, old_norm.as_str()] {
        if needle.trim().is_empty() {
            continue;
        }
        let mut exact_hits = Vec::new();
        let mut from = 0usize;
        while let Some(rel_pos) = region[from..].find(needle) {
            let raw_start = region_start_b + from + rel_pos;
            let raw_end = raw_start + needle.len();
            let line_start = src[..raw_start].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let indent = line_indent(&src[line_start..]);
            let lead_is_ws = src[line_start..raw_start]
                .chars()
                .all(|c| c == ' ' || c == '\t');
            let at_eol = raw_end == src.len()
                || src[raw_end..].starts_with('\n')
                || src[raw_end..].starts_with("\r\n");
            // Never slice a Python statement header in half (`from os import PathLike, stat` matching a prefix of
            // `from os import PathLike, stat, BrokenPipe` and leaving `, BrokenPipe` glued onto `new_string`).
            if lead_is_ws
                && !at_eol
                && !needle.contains('\n')
                && is_python_statement_prefix(needle.trim_start())
            {
                from += rel_pos + needle.len().max(1);
                continue;
            }
            let (abs_start, abs_end) = if lead_is_ws && (at_eol || needle.contains('\n')) {
                let snapped_end = if src[raw_end..].starts_with("\r\n") {
                    raw_end + 2
                } else if src[raw_end..].starts_with('\n') {
                    raw_end + 1
                } else {
                    raw_end
                };
                (line_start, snapped_end)
            } else {
                (raw_start, raw_end)
            };
            let s_ln = byte_to_line_1based(abs_start);
            let e_ln = byte_to_line_1based(abs_end.saturating_sub(1));
            exact_hits.push((abs_start, abs_end, indent, s_ln, e_ln));
            from += rel_pos + needle.len().max(1);
        }
        if !exact_hits.is_empty() {
            return exact_hits;
        }
    }

    Vec::new()
}

/// `new` (2+ non-blank lines, compared trimmed line by line) already occurs as a contiguous
/// block in `live_src`.
pub fn already_applied(live_src: &str, new: &str) -> bool {
    let want: Vec<&str> = new
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if want.len() < 2 {
        return false;
    }
    let have: Vec<&str> = live_src
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    have.windows(want.len()).any(|w| w == want.as_slice())
}

/// One-line form of [`already_applied`]: the single `old` line is gone, the `new` line (an edit of
/// it: shared prefix of at least half its length) is in the file.
pub fn already_applied_line(live_src: &str, old: &str, new: &str) -> bool {
    let one = |s: &str| -> Option<String> {
        let mut it = s.lines().map(str::trim).filter(|l| !l.is_empty());
        let first = it.next()?;
        it.next().is_none().then(|| first.to_string())
    };
    let (Some(o), Some(n)) = (one(old), one(new)) else {
        return false;
    };
    if o == n || n.len() < 16 {
        return false;
    }
    let shared = o.chars().zip(n.chars()).take_while(|(a, b)| a == b).count();
    if shared * 2 < o.len().min(n.len()) {
        return false;
    }
    let lines: Vec<&str> = live_src.lines().map(str::trim).collect();
    lines.contains(&n.as_str()) && !lines.contains(&o.as_str())
}

/// 1-based `(start, end)` window of at most `max_lines` inside `lo..=hi`, centred on the source
/// line closest (edit distance, trimmed) to the first non-blank line of `old`.
pub fn closest_window(
    live_src: &str,
    old: &str,
    lo: usize,
    hi: usize,
    max_lines: usize,
) -> (usize, usize) {
    let lines: Vec<&str> = live_src.lines().collect();
    let hi = hi.min(lines.len()).max(lo);
    let fallback = (lo, hi.min(lo + max_lines));
    let Some(first) = old.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return fallback;
    };
    let Some(anchor) = (lo..=hi)
        .filter(|&n| n >= 1 && !lines[n - 1].trim().is_empty())
        .min_by_key(|&n| line_edit_distance(lines[n - 1].trim(), first))
    else {
        return fallback;
    };
    let start = anchor.saturating_sub(3).max(lo);
    (start, hi.min(start + max_lines))
}

/// The `old`-sized window of lines `lo..=hi` that most of `old`'s lines resemble (equal after
/// trimming, or within a fifth of their length), when at least half of them do.
pub fn best_line_window(live_src: &str, old: &str, lo: usize, hi: usize) -> Option<(usize, usize)> {
    let lines: Vec<&str> = live_src.lines().collect();
    let old_lines: Vec<&str> = old.trim_end_matches('\n').lines().collect();
    let n = old_lines.len();
    if n == 0 || lo == 0 {
        return None;
    }
    let close = |a: &str, b: &str| {
        let (a, b) = (
            strip_trailing_comment(a.trim()),
            strip_trailing_comment(b.trim()),
        );
        a == b || line_edit_distance(a, b) <= (a.len().max(b.len()) / 5).max(1)
    };
    let (best, score) = (lo..=hi.min(lines.len()))
        .filter(|&w| w + n - 1 <= lines.len())
        .map(|w| {
            let sc = (0..n)
                .filter(|&k| close(old_lines[k], lines[w - 1 + k]))
                .count();
            (w, sc)
        })
        .max_by_key(|&(w, sc)| (sc, std::cmp::Reverse(w)))?;
    (score * 2 >= n && score > 0).then_some((best, best + n - 1))
}

/// `line` without a trailing `# comment` (a `#` outside string quotes), right-trimmed.
pub fn strip_trailing_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, ch) in line.char_indices() {
        match quote {
            Some(_) if escaped => escaped = false,
            Some(_) if ch == '\\' => escaped = true,
            Some(q) if ch == q => quote = None,
            Some(_) => {}
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            None if ch == '#' => return line[..i].trim_end(),
            None => {}
        }
    }
    line.trim_end()
}

/// Character edit distance (Levenshtein) between two short lines.
pub fn line_edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Find the single window of `live_src` (inside `line_range` if any matches there, else the whole
/// file) whose lines equal `old`'s lines after trimming, except for 1-2 lines within 2 characters
/// and any lines that differ only by a trailing `# comment`.
/// Returns `(file's window text, new with the same line corrections, note)`; `None` when there is
/// no such window or more than one.
pub fn fuzzy_realign(
    live_src: &str,
    old: &str,
    new: &str,
    line_range: Option<(usize, usize)>,
) -> Option<(String, String, String)> {
    let old_lines: Vec<&str> = old.lines().collect();
    let n = old_lines.len();
    if n < 2 {
        return None;
    }
    let src_lines: Vec<&str> = live_src.lines().collect();
    let scan = |lo: usize, hi: usize| -> Vec<(usize, Vec<usize>, usize)> {
        let mut hits = Vec::new();
        for i in lo..hi.min(src_lines.len()).saturating_sub(n - 1) {
            let mut near = Vec::new();
            let mut comment_only = 0usize;
            let ok = (0..n).all(|k| {
                let (a, b) = (old_lines[k].trim(), src_lines[i + k].trim());
                if a == b {
                    return true;
                }
                // a trailing `# comment` added or dropped in `old_string`
                let (ac, bc) = (strip_trailing_comment(a), strip_trailing_comment(b));
                if !ac.is_empty() && ac == bc {
                    comment_only += 1;
                    return true;
                }
                let d = line_edit_distance(a, b);
                // quote style only: a docstring retyped as `"""..."""` where the file has `"..."`
                let quotes = |s: &str| {
                    s.replace("\"\"\"", "\"")
                        .replace("'''", "\"")
                        .replace('\'', "\"")
                };
                if d == 1 || (a.len().max(b.len()) >= 4 && d <= 2) || quotes(a) == quotes(b) {
                    near.push(k);
                    return true;
                }
                false
            });
            if ok
                && near.len() <= 2
                && near.len() + comment_only >= 1
                && near.len() + comment_only < n
            {
                hits.push((i, near, comment_only));
            }
        }
        hits
    };
    let mut hits = line_range
        .map(|(s, e)| scan(s.saturating_sub(1), e))
        .unwrap_or_default();
    if hits.is_empty() {
        hits = scan(0, src_lines.len());
    }
    if hits.len() != 1 {
        return None;
    }
    let (i, near, comment_only) = hits.remove(0);
    let mut fixed_old = src_lines[i..i + n].join("\n");
    if old.ends_with('\n') {
        fixed_old.push('\n');
    }
    let mut fixed_new = new.to_string();
    let mut notes = Vec::new();
    if comment_only > 0 {
        notes.push(format!(
            "{comment_only} line(s) matched ignoring a trailing `# comment`"
        ));
    }
    for k in near {
        let (slip, real) = (old_lines[k].trim(), src_lines[i + k].trim());
        notes.push(format!(
            "old_string line {} `{slip}` read as file's `{real}`",
            k + 1
        ));
        fixed_new = fixed_new
            .split('\n')
            .map(|l| {
                if l.trim() == slip {
                    format!("{}{real}", &l[..l.len() - l.trim_start().len()])
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    Some((
        fixed_old,
        fixed_new,
        format!("auto-corrected: {}", notes.join(", ")),
    ))
}

fn find_snippet_matches_with_unescape(
    src: &str,
    clean_old: &str,
    line_range: Option<(usize, usize)>,
) -> Vec<(usize, usize, isize, usize, usize)> {
    let direct = find_all_snippet_matches(src, clean_old, line_range);
    if !direct.is_empty() {
        return direct;
    }
    let unesc = unescape_once(clean_old);
    if unesc != clean_old {
        return find_all_snippet_matches(src, &unesc, line_range);
    }
    Vec::new()
}

/// Bridge edits when the model copied header lines + executable body lines across a docstring
/// omitted by `view` (`--- ... docstring lines A..B omitted ... ---`), preserving the docstring intact.
/// Old text taken from the ORIGINAL file after an earlier edit changed that region:
/// strip the lines old and new
/// share at both ends; replace the changed old lines with the new ones in `live` when they occur exactly once
/// there, or for a pure insertion insert after the last shared line when that line occurs exactly once.
pub fn rebase_stale_edit(live: &str, old: &str, new: &str) -> Option<String> {
    let o: Vec<&str> = old.lines().collect();
    let n: Vec<&str> = new.lines().collect();
    let mut pre = 0;
    while pre < o.len() && pre < n.len() && o[pre] == n[pre] {
        pre += 1;
    }
    let mut suf = 0;
    while suf < o.len() - pre && suf < n.len() - pre && o[o.len() - 1 - suf] == n[n.len() - 1 - suf]
    {
        suf += 1;
    }
    let old_mid = &o[pre..o.len() - suf];
    let new_mid = &n[pre..n.len() - suf];
    // whole-line matching only: a substring hit would slice a longer line in half
    let lines: Vec<&str> = live.split('\n').collect();
    let windows = |want: &[&str]| -> Vec<usize> {
        (0..lines.len().saturating_sub(want.len() - 1))
            .filter(|&i| (0..want.len()).all(|k| lines[i + k].trim_end() == want[k].trim_end()))
            .collect()
    };
    let (at, cut) = if old_mid.iter().all(|l| l.trim().is_empty()) {
        // pure insertion after the last shared line
        let anchor = *o.get(pre.checked_sub(1)?)?;
        if anchor.trim().is_empty() || new_mid.iter().all(|l| l.trim().is_empty()) {
            return None;
        }
        let hits = windows(&[anchor]);
        let [i] = hits[..] else { return None };
        (i + 1, 0)
    } else {
        let hits = windows(old_mid);
        let [i] = hits[..] else { return None };
        (i, old_mid.len())
    };
    let mut out: Vec<&str> = lines[..at].to_vec();
    out.extend_from_slice(new_mid);
    out.extend_from_slice(&lines[at + cut..]);
    Some(out.join("\n"))
}

/// Text compared modulo copying slips in string-heavy code: adjacent literals joined across a line
/// break (`r"a"` / `r"b"` written as `r"ab"`), spacing, and doubled backslashes (`\\.` for `\.`).
fn loose_code(s: &str) -> String {
    let joins = Regex::new(r#"(["'])\s*\n\s*[rRbBuUfF]{0,2}["']"#).expect("regex");
    let joined = joins.replace_all(s, "");
    let mut out = String::new();
    let mut prev_bs = false;
    for c in joined.chars().filter(|c| !c.is_whitespace()) {
        if c == '\\' && prev_bs {
            continue;
        }
        prev_bs = c == '\\';
        out.push(c);
    }
    out
}

/// Old text that merged adjacent string lines into one or doubled its backslashes:
/// find the one window of whole live lines (non-blank
/// at both ends) equal to `old` under [`loose_code`] and replace it with `new`, undoing doubled backslashes in `new`
/// when the window has none and shifting `new` to the window's indentation.
pub fn loose_realign(live: &str, old: &str, new: &str) -> Option<String> {
    let want = loose_code(old);
    if want.len() < 12 {
        return None;
    }
    let lines: Vec<&str> = live.split('\n').collect();
    let mut hits = Vec::new();
    for i in 0..lines.len() {
        if lines[i].trim().is_empty() {
            continue;
        }
        for j in i..lines.len() {
            if lines[j].trim().is_empty() {
                continue;
            }
            let got = loose_code(&lines[i..=j].join("\n"));
            if got == want {
                hits.push((i, j));
            }
            if got.len() >= want.len() {
                break;
            }
        }
    }
    let [(i, j)] = hits[..] else { return None };
    let window = lines[i..=j].join("\n");
    let mut new_text = if old.contains("\\\\") && !window.contains("\\\\") {
        new.replace("\\\\", "\\")
    } else {
        new.to_string()
    };
    let first_old = old.lines().find(|l| !l.trim().is_empty())?;
    let delta = line_indent(lines[i]) - line_indent(first_old);
    if delta != 0 {
        new_text = shift_indent(&new_text, delta);
    }
    let mut out: Vec<&str> = lines[..i].to_vec();
    out.extend(new_text.trim_end_matches('\n').split('\n'));
    out.extend_from_slice(&lines[j + 1..]);
    Some(out.join("\n"))
}

/// Lines before the first docstring, the docstring itself, and the lines after it (`None` without one).
fn split_at_docstring(text: &str) -> Option<(Vec<&str>, Vec<&str>, Vec<&str>)> {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.iter().position(|l| {
        let t = l.trim_start();
        t.starts_with("\"\"\"") || t.starts_with("'''") || t.starts_with("r\"\"\"")
    })?;
    let q = if lines[start].contains("'''") {
        "'''"
    } else {
        "\"\"\""
    };
    let end = if lines[start].matches(q).count() >= 2 {
        start
    } else {
        start + 1 + lines[start + 1..].iter().position(|l| l.contains(q))?
    };
    Some((
        lines[..start].to_vec(),
        lines[start..=end].to_vec(),
        lines[end + 1..].to_vec(),
    ))
}

/// `old` and `new` carry the SAME docstring but it is not the file's (an abbreviated one). Apply
/// the code before and after the docstring as two separate replacements, each only if its old text occurs
/// exactly once in the file; the file's own docstring (and any doctest in it) stays.
pub fn edit_around_docstring(live: &str, old: &str, new: &str) -> Option<String> {
    let (po, dold, qo) = split_at_docstring(old)?;
    let (pn, dnew, qn) = split_at_docstring(new)?;
    if dold != dnew {
        return None;
    }
    let trim = |v: &[&str]| -> String {
        let s = v
            .iter()
            .position(|l| !l.trim().is_empty())
            .unwrap_or(v.len());
        let e = v
            .iter()
            .rposition(|l| !l.trim().is_empty())
            .map_or(s, |i| i + 1);
        v[s..e.max(s)].join("\n")
    };
    let mut out = live.to_string();
    let mut applied = 0;
    for (a, b) in [(trim(&po), trim(&pn)), (trim(&qo), trim(&qn))] {
        if a == b {
            continue;
        }
        if a.trim().is_empty() || out.matches(&a).count() != 1 {
            return None;
        }
        out = out.replacen(&a, &b, 1);
        applied += 1;
    }
    (applied > 0).then_some(out)
}

fn try_splice_across_omitted_docstring(
    rel_file: &str,
    live_src: &str,
    defs: &[crate::ast_graph::DefNode],
    clean_old: &str,
    clean_new: &str,
) -> Result<Option<(String, String)>> {
    let strip_omitted_banner = |txt: &str| -> Vec<String> {
        txt.lines()
            .filter(|l| {
                let t = l.trim();
                !(t.starts_with("--- ") && t.ends_with(" ---") && t.contains("omitted"))
            })
            .map(ToString::to_string)
            .collect()
    };
    let old_lines = strip_omitted_banner(clean_old);
    let new_lines = strip_omitted_banner(clean_new);
    if old_lines.len() < 2 {
        return Ok(None);
    }
    let src_lines: Vec<&str> = live_src.split_inclusive('\n').collect();

    for d in defs {
        if d.body_start_line <= d.start_line + 15 {
            continue;
        }
        // Try splitting `old_lines` into `old_lines[..k]` (matching near `d.start_line..d.body_start_line`)
        // and `old_lines[k..]` (matching near `d.body_start_line..=d.end_line`)
        for k in 1..old_lines.len() {
            let head_part = old_lines[..k].join("\n");
            let body_part = old_lines[k..].join("\n");
            let head_hits = find_all_snippet_matches(
                live_src,
                &head_part,
                Some((d.start_line, d.body_start_line.saturating_sub(1))),
            );
            let body_hits = find_all_snippet_matches(
                live_src,
                &body_part,
                Some((d.body_start_line.saturating_sub(3), d.end_line)),
            );
            if head_hits.len() == 1 && body_hits.len() == 1 {
                let (_, _, _, h_s, h_e) = head_hits[0];
                let (_, _, _, b_s, b_e) = body_hits[0];
                if h_e + 1 < b_s {
                    let preserved_doc = src_lines[h_e..b_s.saturating_sub(1)].concat();
                    // Split `new_lines` at the corresponding header end (`k` lines if unchanged header)
                    let split_new = k.min(new_lines.len());
                    let new_head = new_lines[..split_new].join("\n");
                    let new_body = new_lines[split_new..].join("\n");
                    let mut combined_mid = String::new();
                    if !new_head.is_empty() {
                        combined_mid.push_str(&new_head);
                        combined_mid.push('\n');
                    }
                    combined_mid.push_str(&preserved_doc);
                    if !new_body.is_empty() {
                        combined_mid.push_str(&new_body);
                        combined_mid.push('\n');
                    }
                    let cand = format!(
                        "{}{}{}",
                        src_lines[..h_s.saturating_sub(1)].concat(),
                        combined_mid,
                        src_lines[b_e..].concat()
                    );
                    if check_python_syntax(&cand).is_none() {
                        return Ok(Some((
                            cand,
                            format!(
                                "updated `{}` in `{rel_file}`:{h_s}-{b_e} (preserved omitted docstring lines {}..{})",
                                d.qualname,
                                h_e + 1,
                                b_s - 1
                            ),
                        )));
                    }
                }
            }
        }
    }
    Ok(None)
}

fn match_old_snippet(src: &str, old: &str) -> Option<(usize, usize, isize)> {
    find_all_snippet_matches(src, old, None)
        .into_iter()
        .next()
        .map(|(s, e, ind, _, _)| (s, e, ind))
}

fn insert_imports(src: &str, imp_lines: &str) -> String {
    let existing: BTreeSet<&str> = src.lines().map(str::trim).collect();
    let to_add: Vec<&str> = imp_lines
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !existing.contains(*l))
        .collect();
    if to_add.is_empty() {
        return src.to_string();
    }
    let lines: Vec<&str> = src.split_inclusive('\n').collect();
    let mut insert_at = 0usize;
    for (i, ln) in lines.iter().enumerate() {
        let t = ln.trim();
        if t.starts_with("import ") || t.starts_with("from ") {
            insert_at = i + 1;
        }
    }
    let block = format!("{}\n", to_add.join("\n"));
    format!(
        "{}{}{}",
        lines[..insert_at].concat(),
        block,
        lines[insert_at..].concat()
    )
}

fn insert_method_into_class(
    full_path: &Path,
    rel_file: &str,
    live_src: &str,
    cls_def: &crate::ast_graph::DefNode,
    clean_new: &str,
    meth_name: &str,
) -> Result<(String, String)> {
    let lines: Vec<&str> = live_src.split_inclusive('\n').collect();
    let cut = cls_def.end_line.min(lines.len());
    let cls_indent = lines
        .get(cls_def.def_line.saturating_sub(1))
        .or_else(|| lines.get(cls_def.start_line.saturating_sub(1)))
        .map(|l| line_indent(l))
        .unwrap_or(0);
    let target_indent = Some(cls_indent + 4);
    let dedented = dedent_block(clean_new);
    let indented_method = shift_indent(&dedented, cls_indent + 4);

    let assemble = |candidate: &str| -> String {
        let mut c = candidate.to_string();
        if !c.is_empty() && !c.ends_with('\n') {
            c.push('\n');
        }
        let prefix = lines[..cut].concat();
        let sep = if prefix.ends_with("\n\n") || prefix.is_empty() {
            ""
        } else {
            "\n"
        };
        format!("{prefix}{sep}{c}{}", lines[cut..].concat())
    };
    let initial = assemble(&indented_method);
    let (final_src, note) =
        auto_repair_and_validate(initial, &indented_method, target_indent, assemble).map_err(
            |e| {
                anyhow::anyhow!(
                    "{rel_file} (insert method {meth_name} into {}): {e}",
                    cls_def.qualname
                )
            },
        )?;
    crate::write_lf(full_path, &final_src)?;
    let note_tag = if note.is_empty() {
        String::new()
    } else {
        format!(" ({note})")
    };
    Ok((
        rel_file.to_string(),
        format!(
            "inserted new method `{}.{meth_name}` into class `{}` in `{rel_file}`{note_tag}",
            cls_def.qualname, cls_def.qualname
        ),
    ))
}

pub(crate) fn record_created_file(env: &EnvPaths, rel_file: &str) {
    let p = env.agent_dir.join("created_files.json");
    let mut list: Vec<String> = crate::read_lf(&p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if !list.contains(&rel_file.to_string()) {
        list.push(rel_file.to_string());
        let _ = crate::write_lf(&p, &serde_json::to_string(&list).unwrap_or_default());
    }
    let _ = Command::new("git")
        .args(["add", "-N", "--", rel_file])
        .current_dir(&env.ws)
        .output();
}

pub(crate) fn git_show_head(ws: &Path, rel_file: &str) -> String {
    Command::new("git")
        .args(["show", &format!("HEAD:{rel_file}")])
        .current_dir(ws)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).replace("\r\n", "\n"))
        .unwrap_or_default()
}

/// Count `(added_lines, deleted_lines)` in a unified git diff.
pub fn diff_line_counts(diff_text: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut deleted = 0usize;
    for ln in diff_text.lines() {
        if ln.starts_with("+++ ") || ln.starts_with("--- ") {
            continue;
        }
        if ln.starts_with('+') {
            added += 1;
        } else if ln.starts_with('-') {
            deleted += 1;
        }
    }
    (added, deleted)
}

/// Return `true` if a unified diff represents a destructive mass deletion (`deleted > 60` and `deleted > added * 4`).
pub fn is_destructive_diff(diff_text: &str) -> bool {
    let (added, deleted) = diff_line_counts(diff_text);
    deleted > 60 && deleted > added.saturating_mul(4)
}

/// Save the current `git diff HEAD` to `/tmp/.agent/best.patch` whenever `/workspace` has a valid,
/// non-destructive compiling diff.
pub fn save_best_patch_snapshot(env: &EnvPaths) -> usize {
    if let Ok(out) = Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "HEAD"])
        .current_dir(&env.ws)
        .output()
        && out.status.success()
        && !out.stdout.is_empty()
    {
        let diff_str = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        if !is_destructive_diff(&diff_str) {
            let best_path = env.agent_dir.join("best.patch");
            let _ = crate::write_lf(&best_path, &diff_str);
        }
        return diff_str.len();
    }
    0
}

#[derive(Debug, Clone, Default)]
pub struct StaticCriticReport {
    /// High-confidence regressions (newly introduced undefined names, missing `self.<method>`,
    /// or deleted existing public functions/classes/methods). Blocks `READY` in `verify`.
    pub blocking: Vec<String>,
    /// Advisory signature-change notes for top-level non-builtin functions.
    pub advisory: Vec<String>,
}

impl StaticCriticReport {
    pub fn is_empty(&self) -> bool {
        self.blocking.is_empty() && self.advisory.is_empty()
    }
}

/// Run the Static Critic across all touched files:
/// - `blocking`: undefined names, missing `self` methods, accidental deletion of existing public defs
/// - `advisory`: changed signatures vs callers (`stale_callers`) and graph `untouched_callers`
pub fn run_static_critic(env: &EnvPaths, touched_files: &[String]) -> StaticCriticReport {
    let mut blocking = Vec::new();
    let mut advisory = Vec::new();
    let index = load_or_build_index(env).ok();
    let issue_text = crate::read_lf(&env.agent_dir.join("issue.txt")).unwrap_or_default();

    for rel_file in touched_files {
        if !rel_file.ends_with(".py") {
            continue;
        }
        let full_path = crate::join_rel(&env.ws, rel_file);
        let Ok(new_src) = crate::read_lf(&full_path) else {
            continue;
        };
        let base_path = crate::join_rel(&env.agent_dir.join("base"), rel_file);
        let old_src =
            crate::read_lf(&base_path).unwrap_or_else(|_| git_show_head(&env.ws, rel_file));

        // 1. Blocking: newly introduced undefined names, missing `self.<method>`, or deleted public defs
        for item in check_new_undefined_and_self_methods(rel_file, &old_src, &new_src) {
            if !blocking.contains(&item) {
                blocking.push(item);
            }
        }
        for item in check_deleted_public_defs(rel_file, &old_src, &new_src, &issue_text)
            .into_iter()
            .chain(crate::ast_graph::check_property_toggles(
                rel_file,
                &old_src,
                &new_src,
                &issue_text,
            ))
        {
            if !blocking.contains(&item) {
                blocking.push(item);
            }
        }
        let old_nested = crate::ast_graph::find_misindented_nested_methods(&old_src);
        for (line, mname, outer) in crate::ast_graph::find_misindented_nested_methods(&new_src) {
            if !old_nested
                .iter()
                .any(|(_, n, o)| n == &mname && o == &outer)
            {
                let msg = format!(
                    "{rel_file}:{line}: method `def {mname}(self, ...)` is over-indented inside `{outer}()` instead of class body"
                );
                if !blocking.contains(&msg) {
                    blocking.push(msg);
                }
            }
        }
        for item in check_docstring_code_splice(rel_file, &old_src, &new_src) {
            if !blocking.contains(&item) {
                blocking.push(item);
            }
        }
        for item in check_new_imports_runtime(env, rel_file, &old_src, &new_src) {
            if !blocking.contains(&item) {
                blocking.push(item);
            }
        }

        // 2. Advisory: changed function signatures -> check callers across the repo (`stale_callers`)
        let sig_changes = changed_signatures(rel_file, &old_src, &new_src);
        for (qual, fn_name, _old_sig, new_sig) in sig_changes {
            if !new_sig.method
                && !PYTHON_BUILTINS.contains(&fn_name.as_str())
                && !fn_name.starts_with('_')
            {
                for s in check_stale_callers_in_repo(
                    &env.ws,
                    index.as_ref(),
                    rel_file,
                    &fn_name,
                    &new_sig,
                ) {
                    if !advisory.contains(&s) {
                        advisory.push(s);
                    }
                }
            }
            if let Some(idx) = &index {
                for (id, node) in &idx.nodes {
                    if node.file == *rel_file
                        && node.qualname == qual
                        && let Some(callers) = idx.callers.get(id)
                    {
                        let untouched: Vec<String> = callers
                            .iter()
                            .filter_map(|cid| idx.nodes.get(cid))
                            .filter(|cn| {
                                !touched_files.contains(&cn.file) && !is_test_path(&cn.file)
                            })
                            .take(3)
                            .map(|cn| format!("{}:{} ({})", cn.file, cn.start_line, cn.qualname))
                            .collect();
                        if !untouched.is_empty() {
                            let msg = format!(
                                "INFO: signature of `{qual}` changed; check callers in: {}",
                                untouched.join(", ")
                            );
                            if !advisory.contains(&msg) {
                                advisory.push(msg);
                            }
                        }
                    }
                }
            }
        }
    }
    StaticCriticReport { blocking, advisory }
}

/// Detect when Python statements (`def `, `if `, `raise `, `assert `) were accidentally spliced
/// inside an existing multi-line docstring (`"""..."""`) in `old_src`.
fn check_docstring_code_splice(rel_file: &str, old_src: &str, new_src: &str) -> Vec<String> {
    if old_src.is_empty() || new_src.is_empty() {
        return Vec::new();
    }
    let old_lines: Vec<&str> = old_src.lines().collect();
    let new_lines: Vec<&str> = new_src.lines().collect();

    let mut pref = 0usize;
    while pref < old_lines.len() && pref < new_lines.len() && old_lines[pref] == new_lines[pref] {
        pref += 1;
    }
    let line_1based = pref + 1;
    if let Some((_s, str_end, body_hint)) =
        inside_multiline_string(old_src, line_1based, line_1based)
    {
        let mut suff = 0usize;
        while suff < old_lines.len().saturating_sub(pref)
            && suff < new_lines.len().saturating_sub(pref)
            && old_lines[old_lines.len() - 1 - suff] == new_lines[new_lines.len() - 1 - suff]
        {
            suff += 1;
        }
        let hunk_end = new_lines.len().saturating_sub(suff);
        for (offset, &ln) in new_lines[pref..hunk_end].iter().enumerate() {
            let new_ln_1based = pref + offset + 1;
            if inside_multiline_string(new_src, new_ln_1based, new_ln_1based).is_some() {
                let t = ln.trim_start();
                if (t.starts_with("def ")
                    || t.starts_with("async def ")
                    || t.starts_with("raise ")
                    || (t.starts_with("if ") && t.ends_with(':')))
                    && !old_lines.contains(&ln)
                {
                    let where_hint = body_hint
                        .map(|b| format!("executable body starts around line {b}"))
                        .unwrap_or_else(|| format!("docstring closes at line {str_end}"));
                    return vec![format!(
                        "{rel_file}:{new_ln_1based}: code (`{}`) was spliced inside a `\"\"\"...\"\"\"` docstring ({where_hint})! Revert the docstring hunk and place your code in the executable body",
                        t.chars().take(48).collect::<String>()
                    )];
                }
            }
        }
    }
    Vec::new()
}

fn check_stale_callers_in_repo(
    ws: &Path,
    index: Option<&crate::ast_graph::RepoIndex>,
    changed_file: &str,
    fn_name: &str,
    new_sig: &Signature,
) -> Vec<String> {
    let Some(idx) = index else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    let call_re = Regex::new(&format!(
        r"(?:^|[^\w.])({})\(([^()\n]*)\)",
        regex::escape(fn_name)
    ))
    .expect("regex");
    let min_pos = if new_sig.method {
        new_sig.req.saturating_sub(1)
    } else {
        new_sig.req
    };
    let max_pos = if new_sig.var {
        usize::MAX
    } else if new_sig.method {
        new_sig.pos.len().saturating_sub(1)
    } else {
        new_sig.pos.len()
    };

    for f in &idx.files {
        if f == changed_file || is_test_path(f) {
            continue;
        }
        let Ok(txt) = crate::read_lf(&crate::join_rel(ws, f)) else {
            continue;
        };
        if !txt.contains(fn_name) {
            continue;
        }
        for (lineno, line) in txt.lines().enumerate() {
            if line.trim_start().starts_with("def ") || line.trim_start().starts_with("async def ")
            {
                continue;
            }
            if let Some(cap) = call_re.captures(line) {
                let arg_str = cap[2].trim();
                if arg_str.contains('*') {
                    continue;
                }
                let args: Vec<&str> = if arg_str.is_empty() {
                    Vec::new()
                } else {
                    arg_str
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect()
                };
                let mut n_pos = 0usize;
                let mut kw_names = Vec::new();
                for a in args {
                    if let Some((k, _)) = a.split_once('=')
                        && !k.ends_with(['=', '!', '<', '>'])
                    {
                        kw_names.push(k.trim());
                    } else {
                        n_pos += 1;
                    }
                }
                if n_pos > max_pos {
                    warnings.push(format!(
                        "{f}:{}: call `{fn_name}(...)` passes {n_pos} positional args, but `{fn_name}` now accepts at most {max_pos}",
                        lineno + 1
                    ));
                } else if n_pos + kw_names.len() < min_pos {
                    warnings.push(format!(
                        "{f}:{}: call `{fn_name}(...)` passes fewer than {min_pos} required args",
                        lineno + 1
                    ));
                }
                for req_kw in &new_sig.kwreq {
                    if !kw_names.contains(&req_kw.as_str()) {
                        warnings.push(format!(
                            "{f}:{}: call `{fn_name}(...)` is missing required keyword-only arg `{req_kw}=`",
                            lineno + 1
                        ));
                    }
                }
            }
        }
        if warnings.len() >= 3 {
            break;
        }
    }
    warnings
}

/// Execute `/tmp/coder patch` (single or multi-block batch).
pub fn run_patch(env: &EnvPaths, args: &BTreeMap<String, String>, body: &str) -> Result<String> {
    let _guard = MergeLockGuard::acquire(&env.agent_dir);
    let issue_text = crate::read_lf(&env.agent_dir.join("issue.txt")).unwrap_or_default();
    let blocks = parse_patch_blocks(args, body);
    let all_empty = blocks.is_empty()
        || (blocks.len() == 1
            && blocks[0].file.is_empty()
            && blocks[0].symbol.is_none()
            && blocks[0].after_symbol.is_none()
            && blocks[0].old.is_none()
            && blocks[0].start.is_none()
            && blocks[0].after.is_none()
            && blocks[0].imports.is_none()
            && blocks[0].new_text.trim().is_empty());
    if all_empty {
        return Ok(
            "USAGE:\n  1. Edit lines inside symbol/file: `{\"file\": \"pkg/mod.py\", \"symbol\": \"Class.method\", \"old_string\": \"...\", \"new_string\": \"...\"}`\n  2. Add/replace whole method or function: `{\"file\": \"pkg/mod.py\", \"symbol\": \"Class.new_method\", \"body\": \"def new_method(self, ...):\\n    ...\"}`"
                .to_string(),
        );
    }

    let mut applied_msgs = Vec::new();
    let mut errors = Vec::new();
    let mut touched_files = Vec::new();

    for b in &blocks {
        match apply_single_block(env, b, &issue_text) {
            Ok((f, msg)) => {
                if !touched_files.contains(&f) {
                    touched_files.push(f);
                }
                applied_msgs.push(format!("  + {msg}"));
            }
            Err(e) => {
                errors.push(format!("  ! {e}"));
            }
        }
    }

    // Invalidate cached index so subsequent `view` / `verify` sees updated line numbers & signatures
    if !touched_files.is_empty() {
        let _ = std::fs::remove_file(env.agent_dir.join("index.json"));
        crate::record_recent_files(env, &touched_files);
    }

    let patch_bytes = save_best_patch_snapshot(env);
    let critic_report = run_static_critic(env, &touched_files);

    // Re-run `/tmp/.agent/repro.py` automatically if captured
    let mut repro_line = String::new();
    let repro_script = env.agent_dir.join("repro.py");
    if !touched_files.is_empty() && repro_script.is_file() {
        let probe_dir = env.agent_dir.join("probe");
        let cmd = format!("python3 \"{}\"", repro_script.display());
        let limit = crate::verify::repro_limit(env, 12);
        let (rc, out) = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, limit);
        if rc == 0 {
            repro_line = "REPRO CHECK: PASS (issue reproduction script now exits 0!)\n".to_string();
        } else {
            let tail: String = out
                .lines()
                .rev()
                .take(4)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join(" | ");
            repro_line = format!("REPRO CHECK: STILL FAILS (exit {rc}: {tail})\n");
        }
    }

    let mut out = String::new();
    if !applied_msgs.is_empty() {
        out.push_str(&format!(
            "PATCH APPLIED ({} block(s), live `/workspace` diff = {} bytes, backed up to `best.patch`):\n{}\n",
            applied_msgs.len(),
            patch_bytes,
            applied_msgs.join("\n")
        ));
    }
    if !errors.is_empty() {
        out.push_str(&format!("PATCH ERRORS:\n{}\n", errors.join("\n")));
    }
    if !repro_line.is_empty() {
        out.push_str(&repro_line);
    }
    if critic_report.is_empty() {
        out.push_str("STATIC CRITIC: CLEAN (0 syntax, undefined-name, or signature mismatches)\n");
    } else {
        if !critic_report.blocking.is_empty() {
            out.push_str(&format!(
                "STATIC CRITIC ERRORS:\n  - {}\n",
                critic_report.blocking.join("\n  - ")
            ));
        }
        if !critic_report.advisory.is_empty() {
            out.push_str(&format!(
                "STATIC CRITIC ADVISORY:\n  - {}\n",
                critic_report.advisory.join("\n  - ")
            ));
        }
    }

    if errors.is_empty() && critic_report.blocking.is_empty() {
        out.push_str(
            "NEXT: Call `run_skill_script` with `args={\"repro\": \"import ...\\nassert ...\"}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"` to verify and unlock `submit_patch`.\n",
        );
    } else {
        out.push_str("NEXT: Fix the reported error via `run_skill_script` (`file_path=\"scripts/patch.py\"`, `skill_name=\"swe-coder\"`), then call `scripts/verify.py`.\n");
    }

    Ok(clip_tokens(&out, 550))
}

fn is_compound_suite_header(trimmed: &str) -> bool {
    let code_only = trimmed.split('#').next().unwrap_or(trimmed).trim_end();
    code_only.ends_with(':')
        && [
            "if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except ", "except:",
            "finally:",
        ]
        .iter()
        .any(|kw| code_only.starts_with(kw))
}

fn is_control_flow_terminator(trimmed: &str) -> bool {
    ["raise ", "raise", "return ", "return", "break", "continue"]
        .iter()
        .any(|kw| {
            trimmed == *kw
                || (kw.ends_with(' ') && trimmed.starts_with(kw))
                || trimmed.starts_with(&format!("{kw}#"))
        })
}

fn is_python_statement_prefix(trimmed_start: &str) -> bool {
    [
        "from ",
        "import ",
        "def ",
        "async def ",
        "class ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "with ",
        "try:",
        "except ",
        "except:",
        "finally:",
        "return ",
        "raise ",
        "assert ",
        "@",
    ]
    .iter()
    .any(|kw| trimmed_start.starts_with(kw))
}

fn check_new_imports_runtime(
    env: &EnvPaths,
    rel_file: &str,
    old_src: &str,
    new_src: &str,
) -> Vec<String> {
    let old_lines: BTreeSet<&str> = old_src.lines().map(str::trim).collect();
    let mut candidates: Vec<(usize, String)> = Vec::new();
    for (idx, ln) in new_src.lines().enumerate() {
        let ln_no = idx + 1;
        if line_indent(ln) != 0 || inside_multiline_string(new_src, ln_no, ln_no).is_some() {
            continue;
        }
        let t = ln.trim();
        if old_lines.contains(t)
            || t.starts_with("from .")
            || t.starts_with("from __future__")
            || t.ends_with('\\')
            || delim_delta_line(t) != 0
        {
            continue;
        }
        if t.starts_with("import ") || t.starts_with("from ") {
            candidates.push((ln_no, t.to_string()));
        }
    }
    if candidates.is_empty() || candidates.len() > 6 {
        return Vec::new();
    }
    let mut out_errs = Vec::new();

    // 1. Deterministic stdlib guard: `os` / `io` / `sys` / `errno` / `builtins` never export PascalCase
    // `<Name>` when `<Name>Error` is a Python built-in exception (e.g. `from os import BrokenPipe` -> `BrokenPipeError`;
    // `errno` constants are `UPPER_SNAKE_CASE` like `errno.EPIPE`).
    for (ln_no, stmt) in &candidates {
        if let Some(rest) = stmt.strip_prefix("from ")
            && let Some((mod_part, imp_part)) = rest.split_once(" import ")
            && matches!(
                mod_part.trim(),
                "os" | "io" | "sys" | "errno" | "builtins" | "socket"
            )
        {
            for item in imp_part.split(',') {
                let raw_sym = item
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_matches(['(', ')']);
                if raw_sym
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_uppercase())
                    && raw_sym.chars().any(|c| c.is_ascii_lowercase())
                    && !PYTHON_BUILTINS.contains(&raw_sym)
                {
                    let candidate_err = format!("{raw_sym}Error");
                    if PYTHON_BUILTINS.contains(&candidate_err.as_str()) {
                        out_errs.push(format!(
                            "{rel_file}:{ln_no}: ImportError: cannot import name '{raw_sym}' from '{}' — Python's built-in exception is `{candidate_err}` (use `{candidate_err}` directly without importing `{raw_sym}`!)",
                            mod_part.trim()
                        ));
                    }
                }
            }
        }
    }

    // 2. Direct CPython runtime import verification (no shell quoting issues on any OS)
    let mut py_paths = Vec::new();
    let src_dir = env.ws.join("src");
    if src_dir.is_dir() {
        py_paths.push(src_dir);
    }
    py_paths.push(env.ws.clone());
    if let Ok(existing) = std::env::var("PYTHONPATH")
        && !existing.is_empty()
    {
        py_paths.extend(std::env::split_paths(&existing));
    }
    let py_env =
        std::env::join_paths(&py_paths).unwrap_or_else(|_| env.ws.as_os_str().to_os_string());
    let bins: &[&str] = if cfg!(windows) {
        &["python", "python3"]
    } else {
        &["python3", "python"]
    };

    let name_re = Regex::new(r"cannot import name '([^']+)'").expect("regex");
    let mod_re = Regex::new(r"ModuleNotFoundError: No module named '([^']+)'").expect("regex");
    for (ln_no, stmt) in candidates {
        if out_errs
            .iter()
            .any(|e: &String| e.starts_with(&format!("{rel_file}:{ln_no}:")))
        {
            continue;
        }
        let py_one = format!(
            "import sys\ntry:\n    exec({stmt:?}, {{}})\nexcept (ImportError, ModuleNotFoundError) as e:\n    print(f'{{type(e).__name__}}: {{e}}')\n    sys.exit(3)\n"
        );
        let mut diag: Option<String> = None;
        for &bin in bins {
            let Ok(mut child) = Command::new(bin)
                .args(["-c", &py_one])
                .current_dir(&env.ws)
                .env("PYTHONPATH", &py_env)
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
            else {
                continue;
            };
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            let code = loop {
                match child.try_wait() {
                    Ok(Some(st)) => break st.code(),
                    Ok(None) => {
                        if std::time::Instant::now() >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            break None;
                        }
                        thread::sleep(Duration::from_millis(15));
                    }
                    Err(_) => break None,
                }
            };
            if code == Some(0) {
                break;
            }
            if code == Some(3)
                && let Ok(out) = child.wait_with_output()
            {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Some(first) = text.lines().map(str::trim).find(|l| !l.is_empty()) {
                    diag = Some(first.to_string());
                }
                break;
            }
        }
        let Some(err_line) = diag else {
            continue;
        };
        if err_line.starts_with("ImportError:") {
            let mut extra = String::new();
            if let Some(cap) = name_re.captures(&err_line) {
                let bad = &cap[1];
                let candidate_err = format!("{bad}Error");
                if PYTHON_BUILTINS.contains(&candidate_err.as_str()) {
                    extra = format!(
                        " — Python's built-in exception is `{candidate_err}` (use `{candidate_err}` directly without importing `{bad}`!)"
                    );
                }
            }
            out_errs.push(format!("{rel_file}:{ln_no}: {err_line}{extra}"));
        } else if let Some(cap) = mod_re.captures(&err_line) {
            let root_pkg = cap[1].split('.').next().unwrap_or("");
            let is_repo_internal = !root_pkg.is_empty()
                && (env.ws.join(root_pkg).exists()
                    || env.ws.join(format!("{root_pkg}.py")).exists()
                    || env.ws.join("src").join(root_pkg).exists()
                    || env.ws.join("src").join(format!("{root_pkg}.py")).exists());
            if is_repo_internal {
                out_errs.push(format!("{rel_file}:{ln_no}: {err_line}"));
            }
        }
    }
    out_errs
}
