//! `swe-coder` (`/tmp/coder`): the native engine behind the agent's single tool call.
//!
//! Implements five deterministic commands:
//! 1. `debug` (alias `locate`): AST index, call/inheritance graph, mean-centred 256-d structural embeddings,
//!    five-signal `ALSO-CHANGE` counterpart detector, reproducer run and preloaded numbered code slices.
//! 2. `view` (alias `read`): AST symbol (`def`, `async def`, `class`, `Class.method`),
//!    line-range and pattern reader with a hard token cap.
//! 3. `patch` (alias `edit`): single or multi-file/multi-symbol patcher (`=== file=... ===`,
//!    `<<<OLD ... === ... >>>`, `symbol=`, `old=`, `start=/end=`) with gutter stripping, indent repair,
//!    syntax gate, 3-way merge (`git merge-file` + shifted-span splice against `/tmp/.agent/base/`),
//!    write-through `/tmp/.agent/best.patch` backup and static critic.
//! 4. `verify` (aliases `test`, `finalize`): runs the reproducer, the graph-paired `pytest` targets
//!    (`tests_of`), the static critic and the workspace hygiene gate in parallel.
//! 5. `probe` (alias `run`): 15-second Python snippet runner in `/tmp/.agent/probe/`
//!    (`PYTHONPATH=/workspace`) that never writes into `/workspace`.

pub mod ast_graph;
pub mod debug;
pub mod patch;
pub mod verify;
pub mod view;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::Result;
use regex::Regex;

use crate::ast_graph::{RepoIndex, build_repo_index};

/// Read a UTF-8 file and normalize all line endings to LF (`\n`).
pub fn read_lf(path: &Path) -> std::io::Result<String> {
    std::fs::read_to_string(path).map(|s| s.replace("\r\n", "\n"))
}

/// Write a UTF-8 string to disk with LF (`\n`) line endings everywhere.
pub fn write_lf(path: &Path, content: &str) -> std::io::Result<()> {
    let lf = content.replace("\r\n", "\n");
    std::fs::write(path, lf)
}

/// Extract normal path segments (`Component::Normal`) from any path string without hardcoded separators.
pub fn path_segments(raw: &str) -> Vec<String> {
    Path::new(raw)
        .components()
        .filter_map(|c| match c {
            Component::Normal(os) => os.to_str().map(String::from),
            _ => None,
        })
        .collect()
}

/// Join a repo-relative key or path string onto a base `Path` using OS-agnostic `Component::Normal` segments.
pub fn join_rel(base: &Path, rel: &str) -> PathBuf {
    Path::new(rel)
        .components()
        .fold(base.to_path_buf(), |acc, c| match c {
            Component::Normal(os) => acc.join(os),
            _ => acc,
        })
}

/// Normalize a user-supplied file argument into a clean repo-relative path string (stripping any leading
/// `workspace` or `ws` root prefix) using `Component::Normal` segments.
pub fn clean_rel_arg(raw: &str) -> String {
    let mut segs = path_segments(raw.trim());
    if segs.first().map(String::as_str) == Some("workspace") && segs.len() > 1 {
        segs.remove(0);
    }
    PathBuf::from_iter(segs.iter())
        .components()
        .filter_map(|c| match c {
            Component::Normal(os) => os.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn root_subdir(name: &str) -> PathBuf {
    PathBuf::from(std::path::MAIN_SEPARATOR_STR).join(name)
}

/// Resolved workspace (`/workspace`) and scratch (`/tmp` + `/tmp/.agent`) paths.
/// Discovery works both in a container (`/workspace`, `/tmp`) and in a subprocess sandbox
/// (`<sandbox_root>/workspace`, `TMPDIR`).
#[derive(Debug, Clone)]
pub struct EnvPaths {
    pub ws: PathBuf,
    pub tmp: PathBuf,
    pub agent_dir: PathBuf,
}

impl EnvPaths {
    pub fn discover() -> Self {
        let tmp = discover_tmp_dir();
        let ws = discover_workspace_dir(&tmp);
        let agent_dir = tmp.join(".agent");
        let _ = std::fs::create_dir_all(&agent_dir);
        let _ = std::fs::create_dir_all(agent_dir.join("base"));
        let _ = std::fs::create_dir_all(agent_dir.join("probe"));
        Self { ws, tmp, agent_dir }
    }

    pub fn with_roots(ws: PathBuf, tmp: PathBuf) -> Self {
        let agent_dir = tmp.join(".agent");
        let _ = std::fs::create_dir_all(&agent_dir);
        let _ = std::fs::create_dir_all(agent_dir.join("base"));
        let _ = std::fs::create_dir_all(agent_dir.join("probe"));
        Self { ws, tmp, agent_dir }
    }
}

fn discover_tmp_dir() -> PathBuf {
    for var in ["SWE_TMP", "TMPDIR", "TEMP", "TMP"] {
        if let Ok(val) = std::env::var(var) {
            let p = PathBuf::from(val.trim());
            if p.is_dir() {
                return p;
            }
        }
    }
    let sys_tmp = root_subdir("tmp");
    if sys_tmp.is_dir() {
        return sys_tmp;
    }
    std::env::temp_dir()
}

fn discover_workspace_dir(tmp: &Path) -> PathBuf {
    // 1. Explicit `SWE_WS` override
    if let Ok(val) = std::env::var("SWE_WS") {
        let p = PathBuf::from(val.trim());
        if p.is_dir() {
            return p;
        }
    }
    // 2. `PYTHONPATH` entry containing `.git` (set by a subprocess sandbox)
    if let Ok(py_path) = std::env::var("PYTHONPATH") {
        for p in std::env::split_paths(&py_path) {
            if p.join(".git").exists() {
                return p;
            }
        }
    }
    // 3. Current working directory (or ancestor) containing `.git`
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = Some(cwd.as_path());
        while let Some(dir) = cur {
            if dir.join(".git").exists() {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
    }
    // 4. Standard container `workspace` root directory
    let sys_ws = root_subdir("workspace");
    if sys_ws.join(".git").exists() || sys_ws.is_dir() {
        return sys_ws;
    }
    // 5. Sibling `workspace` directory next to `TMPDIR`
    let mut anc = tmp.parent();
    for _ in 0..4 {
        if let Some(p) = anc {
            let cand = p.join("workspace");
            if cand.join(".git").exists() || cand.is_dir() {
                return cand;
            }
            anc = p.parent();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Load `/tmp/.agent/index.json` if present, otherwise build the AST/graph/embedding
/// index across `env.ws` and persist it to `/tmp/.agent/index.json`.
pub fn load_or_build_index(env: &EnvPaths) -> Result<RepoIndex> {
    let idx_path = env.agent_dir.join("index.json");
    if idx_path.is_file()
        && let Ok(bytes) = std::fs::read(&idx_path)
        && let Ok(idx) = serde_json::from_slice::<RepoIndex>(&bytes)
    {
        return Ok(idx);
    }
    let idx = build_repo_index(&env.ws)?;
    if let Ok(json_bytes) = serde_json::to_vec(&idx) {
        let _ = std::fs::write(&idx_path, json_bytes);
    }
    Ok(idx)
}

/// Format 1-based inclusive line slice `[start_line..=end_line]` as clean, un-guttered source lines
/// (the line range is carried in the `--- file:start-end ---` header, so the lines can be quoted verbatim).
pub fn format_numbered_slice(content: &str, start_line: usize, end_line: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let s = start_line.max(1).min(lines.len());
    let e = end_line.max(s).min(lines.len());
    let mut out = String::new();
    for line in &lines[(s - 1)..e] {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Enforce hard output token cap (`1 token ~= 4 chars`) by keeping head + tail if exceeded.
pub fn clip_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens.saturating_mul(4);
    if text.len() <= max_chars {
        return text.to_string();
    }
    let head_budget = (max_chars * 3) / 4;
    let tail_budget = max_chars.saturating_sub(head_budget + 80);
    let head_end = floor_char_boundary(text, head_budget);
    let tail_start = ceil_char_boundary(text, text.len().saturating_sub(tail_budget));
    format!(
        "{}\n... [clipped {} chars to stay within {} token budget] ...\n{}",
        &text[..head_end],
        tail_start.saturating_sub(head_end),
        max_tokens,
        &text[tail_start..]
    )
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

/// Parse `key=value` pairs from a string (supporting single-quoted, double-quoted, and unquoted values).
pub fn parse_kv_words(input: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let kv_re = Regex::new(
        r#"([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|([^\s"']+))"#,
    )
    .expect("regex");
    for cap in kv_re.captures_iter(input) {
        let key = cap[1].to_ascii_lowercase().replace('-', "_");
        let val = if let Some(m) = cap.get(2) {
            m.as_str().replace("\\\"", "\"").replace("\\n", "\n")
        } else if let Some(m) = cap.get(3) {
            m.as_str().replace("\\'", "'").replace("\\n", "\n")
        } else if let Some(m) = cap.get(4) {
            m.as_str().to_string()
        } else {
            String::new()
        };
        map.insert(key, val);
    }
    map
}

fn has_unescaped_quote(s: &str, q: char) -> bool {
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == q {
            return true;
        }
    }
    false
}

/// Strip matching outer quotes ONLY when `raw` starts and ends with the same quote character
/// and has no unescaped matching quote inside (never mangles Python literals like `assert f('a') == 'a'`).
pub fn strip_matching_outer_quotes(raw: &str) -> String {
    let clean_nl = raw.trim_matches(['\r', '\n']);
    let trimmed = clean_nl.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            let inner = &trimmed[1..trimmed.len() - 1];
            if !has_unescaped_quote(inner, first as char) {
                return inner.to_string();
            }
        }
    }
    if clean_nl.contains('\n') {
        clean_nl.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Parse CLI `argv` (after binary name) + optional `stdin_str` into `(subcommand, kv_args, heredoc_body)`.
pub fn parse_cli_and_stdin(
    argv: &[String],
    stdin_str: &str,
) -> (String, BTreeMap<String, String>, String) {
    let mut subcmd = String::new();
    let mut args = BTreeMap::new();
    let mut positional = Vec::new();

    let mut idx = 0usize;
    while idx < argv.len() {
        let raw_arg = &argv[idx];
        let t = raw_arg.trim();
        if idx == 0 && !t.contains('=') && !t.starts_with('-') {
            subcmd = t.to_ascii_lowercase();
            idx += 1;
            continue;
        }
        if let Some(flag) = t.strip_prefix("--").or_else(|| t.strip_prefix('-')) {
            if let Some((k, v)) = flag.split_once('=') {
                let key = k.trim().to_ascii_lowercase().replace('-', "_");
                let val = strip_matching_outer_quotes(v);
                args.insert(key, val);
                idx += 1;
                continue;
            }
            let key = flag.trim().to_ascii_lowercase().replace('-', "_");
            if idx + 1 < argv.len() && !argv[idx + 1].trim().starts_with("--") {
                let val = strip_matching_outer_quotes(&argv[idx + 1]);
                args.insert(key, val);
                idx += 2;
                continue;
            } else {
                args.insert(key, "1".to_string());
                idx += 1;
                continue;
            }
        }
        if let Some((k, v)) = raw_arg.split_once('=') {
            let key = k.trim().to_ascii_lowercase().replace('-', "_");
            // code-carrying values keep their leading indentation (flat `old='    x = 1\n...'`):
            // the shell already removed the quotes, so only an inner quoted value is unwrapped
            let code_key = matches!(
                key.as_str(),
                "old" | "new" | "old_string" | "new_string" | "body" | "code" | "repro"
            );
            let val = if code_key && !v.trim_start().starts_with(['"', '\'']) {
                v.trim_matches(['\r', '\n']).to_string()
            } else {
                strip_matching_outer_quotes(v)
            };
            args.insert(key, val);
        } else if !t.is_empty() {
            positional.push(t.to_string());
        }
        idx += 1;
    }

    if subcmd.is_empty() {
        subcmd = args
            .remove("cmd")
            .or_else(|| args.remove("subcommand"))
            .unwrap_or_else(|| "help".to_string())
            .to_ascii_lowercase();
    }

    // Flat call shape: multi-line values passed on one line arrive through the shell with literal
    // `\n` escapes (`old='a\nb'`); decode them for the code-carrying keys when no real newline is present.
    for key in [
        "old",
        "new",
        "old_string",
        "new_string",
        "body",
        "code",
        "repro",
        "test",
    ] {
        if let Some(v) = args.get_mut(key)
            && !v.contains('\n')
            && v.contains("\\n")
        {
            *v = v.replace("\\n", "\n").replace("\\t", "\t");
        }
        // inside shell single quotes `\'` arrives verbatim: unescape it for the code-carrying keys
        if let Some(v) = args.get_mut(key)
            && v.contains("\\'")
        {
            *v = v.replace("\\'", "'");
        }
    }

    // Map convenience positional arguments if the model omitted `key=`
    if !positional.is_empty() {
        match subcmd.as_str() {
            "debug" | "locate" => {
                // `query=foo bar baz` without quotes: the shell splits it; keep every word
                let rest = positional.join(" ");
                args.entry("query".to_string())
                    .and_modify(|q| {
                        q.push(' ');
                        q.push_str(&rest);
                    })
                    .or_insert(rest);
            }
            "view" | "read" => {
                let p0 = &positional[0];
                if looks_like_file_arg(p0) {
                    args.entry("file".to_string()).or_insert_with(|| p0.clone());
                    if positional.len() >= 2 {
                        args.entry("symbol".to_string())
                            .or_insert_with(|| positional[1].clone());
                    }
                } else if looks_like_symbol_arg(p0) {
                    args.entry("symbol".to_string())
                        .or_insert_with(|| p0.clone());
                }
            }
            "patch" | "edit" => {
                let p0 = &positional[0];
                if looks_like_file_arg(p0) {
                    args.entry("file".to_string()).or_insert_with(|| p0.clone());
                    if positional.len() >= 2 && looks_like_symbol_arg(&positional[1]) {
                        args.entry("symbol".to_string())
                            .or_insert_with(|| positional[1].clone());
                    }
                } else if positional.len() == 1 && looks_like_symbol_arg(p0) && p0.contains('.') {
                    args.entry("symbol".to_string())
                        .or_insert_with(|| p0.clone());
                }
            }
            "verify" | "test" | "finalize" => {
                args.entry("test".to_string())
                    .or_insert_with(|| positional.join(","));
            }
            "probe" | "run" => {
                args.entry("code".to_string())
                    .or_insert_with(|| positional.join(" "));
            }
            _ => {}
        }
    }

    // Normalize common parameter aliases (`old_string` / `snippet` -> `old`)
    if matches!(subcmd.as_str(), "patch" | "edit") && !args.contains_key("old") {
        for alias in [
            "old_string",
            "old_str",
            "snippet",
            "search",
            "target",
            "original",
        ] {
            if let Some(v) = args.remove(alias) {
                args.insert("old".to_string(), v);
                break;
            }
        }
    }

    let body = if !stdin_str.is_empty() {
        stdin_str.to_string()
    } else {
        args.remove("body")
            .or_else(|| args.remove("new_string"))
            .or_else(|| args.remove("new_str"))
            .or_else(|| args.remove("new"))
            .or_else(|| args.remove("replacement"))
            .or_else(|| args.remove("replace"))
            .or_else(|| args.remove("content"))
            .or_else(|| args.remove("patch"))
            .or_else(|| args.remove("diff"))
            .or_else(|| {
                matches!(subcmd.as_str(), "patch" | "edit" | "probe" | "run")
                    .then(|| args.remove("code"))
                    .flatten()
            })
            .unwrap_or_default()
    };

    (subcmd, args, body)
}

pub(crate) fn looks_like_file_arg(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() || t.ends_with('.') || t.contains(char::is_whitespace) {
        return false;
    }
    let segs = path_segments(t);
    if segs.len() > 1 {
        return true;
    }
    matches!(
        Path::new(t).extension().and_then(|e| e.to_str()),
        Some(
            "py" | "pyi"
                | "pyx"
                | "pxd"
                | "toml"
                | "cfg"
                | "ini"
                | "json"
                | "yaml"
                | "yml"
                | "md"
                | "rst"
                | "txt"
                | "c"
                | "h"
                | "sh"
        )
    )
}

fn looks_like_symbol_arg(s: &str) -> bool {
    let t = s.trim();
    !t.is_empty()
        && !t.ends_with('.')
        && !t.starts_with('.')
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Record recently inspected or patched repo-relative file paths into `/tmp/.agent/last_targets.json`
/// (most-recent first) so `patch` can auto-resolve omitted `file=` arguments safely.
pub fn record_recent_files(env: &EnvPaths, files: &[String]) {
    if files.is_empty() {
        return;
    }
    let path = env.agent_dir.join("last_targets.json");
    let mut cur = load_recent_files(env);
    for f in files.iter().rev() {
        let clean = clean_rel_arg(f);
        if clean.is_empty() || crate::ast_graph::is_test_path(&clean) {
            continue;
        }
        cur.retain(|x| x != &clean);
        cur.insert(0, clean);
    }
    cur.truncate(12);
    if let Ok(s) = serde_json::to_string(&cur) {
        let _ = write_lf(&path, &s);
    }
}

/// Load recently inspected or patched repo-relative file paths from `/tmp/.agent/last_targets.json`.
pub fn load_recent_files(env: &EnvPaths) -> Vec<String> {
    let path = env.agent_dir.join("last_targets.json");
    read_lf(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

/// Resolve `raw_file` against `env.ws`, supporting unique path-suffix matches (e.g. `routing.py` -> `fastapi/routing.py`).
pub fn resolve_existing_repo_file(env: &EnvPaths, raw_file: &str) -> Option<String> {
    let clean = clean_rel_arg(raw_file);
    if clean.is_empty() {
        return None;
    }
    if join_rel(&env.ws, &clean).is_file() {
        return Some(clean);
    }
    let idx = load_or_build_index(env).ok()?;
    let target_path = Path::new(&clean);
    let matches: Vec<&String> = idx
        .files
        .iter()
        .filter(|f| Path::new(f.as_str()).ends_with(target_path))
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].clone());
    }
    let non_test: Vec<&String> = matches
        .into_iter()
        .filter(|f| !crate::ast_graph::is_test_path(f))
        .collect();
    if non_test.len() == 1 {
        return Some(non_test[0].clone());
    }
    // Check if exactly one recent file matches the suffix
    let recent = load_recent_files(env);
    let recent_hits: Vec<String> = recent
        .into_iter()
        .filter(|f| Path::new(f.as_str()).ends_with(target_path) && join_rel(&env.ws, f).is_file())
        .collect();
    if recent_hits.len() == 1 {
        return Some(recent_hits[0].clone());
    }
    case_insensitive_path(&env.ws, &clean)
}

/// `README.md` for a file named `ReadMe.md`: each segment matched without case against the real listing,
/// accepted only when every segment has exactly one match.
fn case_insensitive_path(ws: &Path, clean: &str) -> Option<String> {
    let mut dir = ws.to_path_buf();
    let mut out = Vec::new();
    for seg in clean.split('/').filter(|s| !s.is_empty()) {
        let low = seg.to_lowercase();
        let hits: Vec<String> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.to_lowercase() == low)
            .collect();
        let [one] = hits.as_slice() else { return None };
        dir.push(one);
        out.push(one.clone());
    }
    dir.is_file().then(|| out.join("/"))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct TaskClock {
    started_unix: u64,
    calls: usize,
    patched_at_call: usize,
    verified_at_call: usize,
    #[serde(default)]
    last_verify_ready: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct LastCallRecord {
    subcmd: String,
    args: BTreeMap<String, String>,
    body: String,
    repeat: usize,
}

/// The first error line of an engine reply with paths, numbers and quoted text removed, so the same
/// mistake with different arguments still matches.
fn error_signature(out: &str) -> Option<String> {
    let line = out
        .lines()
        .find(|l| l.trim_start().starts_with("! ") || l.starts_with("VERDICT: NEEDS FIX"))?;
    let sig = Regex::new(r"`[^`]*`|\d+")
        .expect("regex")
        .replace_all(line.trim(), "_")
        .into_owned();
    Some(sig)
}

/// The same error three calls in a row, even with different arguments.
fn repeated_error_note(env: &EnvPaths, out: &str) -> Option<String> {
    let path = env.agent_dir.join("last_error.json");
    let prev: Option<(String, usize)> = read_lf(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let Some(sig) = error_signature(out) else {
        let _ = std::fs::remove_file(&path);
        return None;
    };
    let n = match prev {
        Some((p, n)) if p == sig => n + 1,
        _ => 1,
    };
    if let Ok(s) = serde_json::to_string(&(sig.clone(), n)) {
        let _ = write_lf(&path, &s);
    }
    (n >= 3).then(|| {
        format!(
            "🛑 SAME ERROR {n} TIMES IN A ROW: this approach is not working. Do not retry it. Read the error once more, then change the approach: `/tmp/coder view` the target to copy its exact current lines, pick a file that exists (`/tmp/coder debug`), or undo and test another hypothesis."
        )
    })
}

fn record_and_check_duplicate_call(
    env: &EnvPaths,
    subcmd: &str,
    args: &BTreeMap<String, String>,
    body: &str,
) -> usize {
    let path = env.agent_dir.join("last_call.json");
    let prev: Option<LastCallRecord> = read_lf(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let repeat = match prev {
        Some(p) if p.subcmd == subcmd && p.args == *args && p.body == body => p.repeat + 1,
        _ => 1,
    };
    let rec = LastCallRecord {
        subcmd: subcmd.to_string(),
        args: args.clone(),
        body: body.to_string(),
        repeat,
    };
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = write_lf(&path, &s);
    }
    repeat
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_and_tick_clock(env: &EnvPaths) -> (TaskClock, u64) {
    let clock_path = env.agent_dir.join("clock.json");
    let now = unix_now_secs();
    let mut clock: TaskClock = read_lf(&clock_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(TaskClock {
            started_unix: now,
            calls: 0,
            patched_at_call: 0,
            verified_at_call: 0,
            last_verify_ready: false,
        });
    if clock.started_unix == 0 {
        clock.started_unix = now;
    }
    clock.calls += 1;
    let elapsed = now.saturating_sub(clock.started_unix);
    if let Ok(s) = serde_json::to_string(&clock) {
        let _ = write_lf(&clock_path, &s);
    }
    (clock, elapsed)
}

fn save_clock(env: &EnvPaths, clock: &TaskClock) {
    let clock_path = env.agent_dir.join("clock.json");
    if let Ok(s) = serde_json::to_string(clock) {
        let _ = write_lf(&clock_path, &s);
    }
}

fn has_active_patch(env: &EnvPaths) -> bool {
    let ws_has_diff = Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "--quiet", "HEAD"])
        .current_dir(&env.ws)
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);
    ws_has_diff
        || env
            .agent_dir
            .join("best.patch")
            .metadata()
            .is_ok_and(|m| m.len() > 0)
}

/// Return the active harness clock scale (`SWE_TIME_SCALE`, clamped to `1.0..=20.0`, default `1.0`).
pub fn active_time_scale() -> f64 {
    std::env::var("SWE_TIME_SCALE")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(1.0, 20.0))
        .unwrap_or(1.0)
}

/// Scale of the session clocks relative to the 288 s reference session: the skill
/// passes `_budget.MAX_TIME_S` (already multiplied by the local time scale) as `SWE_MAX_TIME_S`.
/// Falls back to `SWE_TIME_SCALE` when the variable is absent.
pub fn budget_scale() -> f64 {
    std::env::var("SWE_MAX_TIME_S")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| (v / 288.0).clamp(0.3, 40.0))
        .unwrap_or_else(active_time_scale)
}

/// Tool-call cap of the session (`_budget.MAX_TOOL_CALLS` as `SWE_MAX_TOOL_CALLS`; 26 is the
/// reference the call thresholds below are scaled from).
pub fn max_tool_calls() -> u32 {
    std::env::var("SWE_MAX_TOOL_CALLS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map(|v| v.clamp(8, 200))
        .unwrap_or(26)
}

/// Run a shell command with `PYTHONPATH` prepended with `ws_pythonpath/src` (if present) + `ws_pythonpath`
/// and a hard wall-clock timeout.
pub fn run_shell_with_timeout(
    cmd: &str,
    cwd: &Path,
    ws_pythonpath: &Path,
    timeout_secs: u64,
) -> (i32, String) {
    let mut command = if cfg!(windows) {
        let mut c = Command::new("cmd");
        // On Windows test hosts where `python3` might be `python`, replace leading `python3 ` with `python `
        let win_cmd = if let Some(rest) = cmd.strip_prefix("python3 ") {
            format!("python {rest}")
        } else {
            cmd.to_string()
        };
        c.args(["/C", &win_cmd]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", cmd]);
        c
    };

    let mut py_paths = Vec::new();
    let src_dir = ws_pythonpath.join("src");
    if src_dir.is_dir() {
        py_paths.push(src_dir);
    }
    py_paths.push(ws_pythonpath.to_path_buf());
    if let Ok(existing) = std::env::var("PYTHONPATH")
        && !existing.is_empty()
    {
        py_paths.extend(std::env::split_paths(&existing));
    }
    let py_env = std::env::join_paths(&py_paths)
        .unwrap_or_else(|_| ws_pythonpath.as_os_str().to_os_string());

    command
        .current_dir(cwd)
        .env("PYTHONPATH", py_env)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        // A repro that waits on stdin: close stdin and kill the whole process group on timeout, so a
        // grandchild cannot keep the output pipe open.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);

    let Ok(mut child) = command.spawn() else {
        return (-1, format!("Failed to spawn command: {cmd}"));
    };

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut s) = stdout_pipe.take() {
            let _ = s.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut s) = stderr_pipe.take() {
            let _ = s.read_to_end(&mut buf);
        }
        buf
    });

    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    let exit_code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code().unwrap_or(1),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    // the whole process group, or a grandchild keeps the pipes open
                    #[cfg(unix)]
                    let _ = Command::new("sh")
                        .args(["-c", &format!("kill -9 -{}", child.id())])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                    let _ = child.kill();
                    let _ = child.wait();
                    break 124;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => break 1,
        }
    };

    let out_bytes = out_handle.join().unwrap_or_default();
    let err_bytes = err_handle.join().unwrap_or_default();
    let mut combined = String::from_utf8_lossy(&out_bytes).replace("\r\n", "\n");
    let err_str = String::from_utf8_lossy(&err_bytes).replace("\r\n", "\n");
    if !err_str.is_empty() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&err_str);
    }
    if exit_code == 124 {
        combined.push_str(&format!("\n[TIMEOUT after {timeout_secs}s]"));
    }
    (exit_code, combined)
}

/// Execute `/tmp/coder <subcmd> ...` using explicit `EnvPaths`.
pub fn run_cli_with_env(env: &EnvPaths, argv: &[String], stdin_str: &str) -> Result<String> {
    let (subcmd, args, body) = parse_cli_and_stdin(argv, stdin_str);
    let (mut clock, elapsed_secs) = load_and_tick_clock(env);
    let repeat_count = record_and_check_duplicate_call(env, &subcmd, &args, &body);
    let scratch_moved = verify::move_root_scratch(&env.ws, &env.tmp);
    // a verify killed during its baseline tests left the patch set aside: put it back before anything else
    let stash_restored = verify::restore_baseline_stash(env);

    let t_scale = budget_scale();
    let soft_warn_secs = (225.0 * t_scale) as u64;
    let reserve_secs = (260.0 * t_scale) as u64;
    let max_calls = max_tool_calls() as usize;
    let reserve_calls = max_calls.saturating_sub(2);
    let warn_calls = max_calls * 3 / 5;
    let block_calls = max_calls.saturating_sub(6);

    // Budget reserve short-circuit (scaled by `SWE_TIME_SCALE`):
    // never short-circuit `verify` (`is_verify_cmd`) or `patch` (`is_patch_cmd`); only short-circuit
    // read-only calls in the final 2 calls (`calls >= reserve_calls`) or after `reserve_secs`.
    let is_verify_cmd = matches!(subcmd.as_str(), "verify" | "test" | "finalize");
    let is_patch_cmd = matches!(subcmd.as_str(), "patch" | "edit");
    if !is_verify_cmd
        && !is_patch_cmd
        && (clock.calls >= reserve_calls || elapsed_secs >= reserve_secs)
        && has_active_patch(env)
    {
        verify::ensure_clean_submittable_workspace(env);
        return Ok(format!(
            "FINAL BUDGET RESERVE (call #{}, {}s elapsed): Workspace diff is ready and backed up to `best.patch`.\n\
             Do NOT call `run_skill_script` again — call the top-level `submit_patch` tool (`{{}}`) immediately on your next turn!",
            clock.calls, elapsed_secs
        ));
    }

    let mut out = match subcmd.as_str() {
        "debug" | "locate" => debug::run_debug(env, &args, &body)?,
        "view" | "read" => match (repeat_count >= 2)
            .then(|| view::page_repeated_view(env, &args, repeat_count))
            .transpose()?
            .flatten()
        {
            Some(paged) => paged,
            None => view::run_view(env, &args)?,
        },
        "patch" | "edit" => {
            let res = patch::run_patch(env, &args, &body)?;
            if res.contains("PATCH APPLIED") {
                clock.patched_at_call = clock.calls;
                clock.last_verify_ready = false;
                save_clock(env, &clock);
            }
            res
        }
        "verify" | "test" | "finalize" => {
            // An identical verify right after the last one sees the same workspace: answer from the
            // saved verdict instead of running the tests again.
            let saved = env.agent_dir.join("last_verify.txt");
            let cached = (repeat_count >= 2).then(|| read_lf(&saved).ok()).flatten();
            let res = match cached {
                Some(prev) => format!(
                    "SAME VERIFY, SAME WORKSPACE: the tests were not run again; the previous result stands. Change the code with /tmp/coder patch (or undo the edit) before verifying again.\n{prev}"
                ),
                None => {
                    let r = verify::run_verify(env, &args, &body, elapsed_secs)?;
                    let _ = write_lf(&saved, &r);
                    r
                }
            };
            clock.verified_at_call = clock.calls;
            clock.last_verify_ready = res.contains("VERDICT: READY");
            save_clock(env, &clock);
            res
        }
        "probe" | "run" => verify::run_probe(env, &args, &body)?,
        // `/tmp/coder submit_patch` and its synonyms are not commands: answer how to finish instead of
        // the usage text.
        "submit_patch" | "submit" | "finish" | "done" | "stop" => submit_reply(
            &subcmd,
            std::env::var("SWE_HAS_SUBMIT").map_or(true, |v| v != "0"),
            clock.last_verify_ready,
        ),
        _ => {
            "USAGE: swe-coder <debug|view|patch|verify|probe> [key=val ...]\n\
             Skills:\n\
               1. debug query=\"...\"          — AST/Graph/Embedding localization + Auto-Repro + Preloaded source\n\
               2. view file=a.py symbol=Foo   — Read symbol, line slice, or multi-target `targets=\"...\"`\n\
               3. patch file=a.py symbol=Foo  — Atomic AST-checked single/multi-file patch + 3-way merge\n\
               4. verify [test=tests/...]     — Parallel Repro + Graph Pytest + Static Critic + Hygiene gate\n\
               5. probe                       — 15s sandboxed Python probe in /tmp/.agent/probe"
                .to_string()
        }
    };

    // At the last call an empty workspace gets the last verified `best.patch` back, and the reply
    // says that the call budget ends.
    if clock.calls + 1 >= max_calls && !has_active_patch(env) {
        let (_, restored, _) = crate::verify::ensure_clean_submittable_workspace(env);
        if restored {
            out.push_str("\nHYGIENE: the workspace had no changes; restored your last verified `best.patch`.");
        }
    }
    if clock.calls + 1 >= max_calls && has_active_patch(env) {
        out.push_str(&format!(
            "\n🛑 CALL {} OF {max_calls}: the tool budget ends with your next call. Call `submit_patch` (`{{}}`) NOW: every other tool will fail with BudgetExceeded.",
            clock.calls
        ));
    }
    // Checkpoint at a third of the call budget while nothing is edited yet: break reading loops
    // on one file.
    if clock.calls == max_calls / 3 && !has_active_patch(env) && !is_patch_cmd {
        out.push_str(&format!(
            "\n🧭 CALL {} OF {max_calls}, NO EDIT YET: if the code you are looking for is not in the file you keep viewing, search the whole repo once with `scripts/debug.py` (`args={{\"query\": \"<exact identifier>\"}}`); otherwise edit the function you have found now.",
            clock.calls
        ));
    }
    // Two attempts: save an unverified attempt at half the budget and start over; keep the better one at the end
    let mut attempt_note = String::new();
    if attempts() == 2 {
        let last = read_lf(&env.agent_dir.join("last_verify.txt")).unwrap_or_default();
        if clock.calls == max_calls / 2 && !clock.last_verify_ready && has_active_patch(env) {
            if let Some(n) = checkpoint_attempt(env, verdict_rank(&last), clock.calls) {
                attempt_note = n;
            }
        } else if clock.calls + 1 >= max_calls || elapsed_secs >= reserve_secs {
            let ckpt: usize = std::fs::read_to_string(env.agent_dir.join("attempt1.meta"))
                .ok()
                .and_then(|m| m.split_whitespace().nth(1).and_then(|c| c.parse().ok()))
                .unwrap_or(usize::MAX);
            let cur = if clock.verified_at_call > ckpt {
                verdict_rank(&last)
            } else {
                0
            };
            if let Some(n) = select_attempt(env, cur) {
                attempt_note = n;
            }
        }
    }
    // One checkpoint at half the call budget for sessions that have not converged.
    if !attempt_note.is_empty() {
        out.push_str(&attempt_note);
    } else if clock.calls == max_calls / 2 {
        if !has_active_patch(env) {
            out.push_str(&format!(
                "\n🧭 HALF BUDGET (call {} of {max_calls}): no edit yet. Commit to your best root-cause hypothesis now: view the one function to change, then edit it via `scripts/patch.py`.",
                clock.calls
            ));
        } else if !clock.last_verify_ready {
            out.push_str(&format!(
                "\n🧭 HALF BUDGET (call {} of {max_calls}): your patch is not verified READY. Run `scripts/verify.py` now; if it fails for a reason your edit cannot fix, undo the edit and test the next hypothesis instead of stacking edits.",
                clock.calls
            ));
        }
    }
    if !scratch_moved.is_empty() {
        out.push_str(&format!(
            "\nSCRATCH: moved {} out of /workspace (they would have gone into your patch): {}. Write scratch files under /tmp, never in the repo.",
            scratch_moved.len(),
            scratch_moved.join(", ")
        ));
    }
    if stash_restored {
        out.push_str(
            "\nRESTORED: your last verify was cut off while it ran the tests on the original code; your edit has been put back in /workspace. Keep verify repros short and never wait for input.",
        );
    }
    if let Some(note) = repeated_error_note(env, &out) {
        out = format!("{note}\n{out}");
    }
    if repeat_count >= 2 && !out.starts_with("REPEATED VIEW") {
        out = format!(
            "⚠️ DUPLICATE CALL (#{repeat_count} identical `{subcmd}` call in a row — workspace did NOT change): Do NOT repeat the exact same arguments! Change your `args`/`symbol`/`old_string` or advance to the next step.\n{out}"
        );
    }

    if clock.calls == 1 && matches!(subcmd.as_str(), "view" | "read") {
        out.push_str(
            "\n💡 TIP: You called `view.py` before `scripts/debug.py`. Run `scripts/debug.py` (`query=\"...\"`) if you need cross-repo AST/Call-Graph localization, `ALSO-CHANGE` counterparts, or inline repro execution.",
        );
    }

    let has_blocking_verdict = out.contains("VERDICT: NEEDS FIX")
        || out.contains("VERDICT: SYNTAX ERROR")
        || out.contains("VERDICT: REPRODUCER SYNTAX ERROR")
        || out.contains("VERDICT: EMPTY DIFF")
        || out.contains("PATCH ERRORS:")
        || out.contains("STATIC CRITIC ERRORS:")
        || out.contains("REPRO CHECK: STILL FAILS")
        // NEEDS ISSUE REPRO is a blocking verdict as well
        || out.contains("VERDICT: NEEDS ISSUE REPRO");
    // After a failing verdict with budget left, say how much is left and ask for a revision
    // instead of an early stop.
    if has_blocking_verdict && elapsed_secs + 60 < reserve_secs && clock.calls < block_calls {
        out.push_str(&format!(
            "\n⏳ TIME LEFT: ~{}s, {} calls. A patch that fails its own check scores 0: do NOT call `submit_patch` now. Read the failure, then revise the edit (or undo it and test the next hypothesis).",
            reserve_secs - elapsed_secs,
            block_calls.saturating_sub(clock.calls)
        ));
    }
    if !has_blocking_verdict
        && (clock.calls >= warn_calls || elapsed_secs >= soft_warn_secs)
        && has_active_patch(env)
    {
        if clock.last_verify_ready {
            out.push_str(&format!(
                "\n⏰ BUDGET WARNING (call #{}, {}s elapsed): Verified patch is ready in `/workspace`. Call `submit_patch` now!",
                clock.calls, elapsed_secs
            ));
        } else {
            out.push_str(&format!(
                "\n⏰ BUDGET WARNING (call #{}, {}s elapsed): Unverified patch in `/workspace` — run `scripts/verify.py` now (fix any failure via `scripts/patch.py`), then call `submit_patch` before call #{reserve_calls}!",
                clock.calls, elapsed_secs
            ));
        }
    }

    if std::env::var("SWE_CALL_SHAPE").as_deref() == Ok("cmd") {
        out = flatten_hints(&out);
    }
    Ok(out)
}

/// Attempts per task, from `SWE_ATTEMPTS` (the boot wrapper sets it from the
/// `budget.attempts` knob); 1 = off.
pub fn attempts() -> u32 {
    std::env::var("SWE_ATTEMPTS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map_or(1, |v| v.clamp(1, 2))
}

/// How good a verify result is, for choosing between attempts: READY > NEEDS ISSUE REPRO (tests pass, no repro) >
/// NEEDS FIX > anything else (no verify, empty diff).
pub fn verdict_rank(verify_out: &str) -> u8 {
    if verify_out.contains("VERDICT: READY") {
        3
    } else if verify_out.contains("VERDICT: NEEDS ISSUE") {
        2
    } else if verify_out.contains("VERDICT: NEEDS FIX") {
        1
    } else {
        0
    }
}

fn ws_diff(env: &EnvPaths) -> Option<Vec<u8>> {
    let o = Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "--binary", "HEAD"])
        .current_dir(&env.ws)
        .output()
        .ok()?;
    (o.status.success() && !o.stdout.is_empty()).then_some(o.stdout)
}

fn git_apply(env: &EnvPaths, patch: &Path, reverse: bool) -> bool {
    let mut args = vec!["apply", "--whitespace=nowarn"];
    if reverse {
        args.push("-R");
    }
    let p = patch.display().to_string();
    args.push(&p);
    Command::new("git")
        .args(&args)
        .current_dir(&env.ws)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Save the live diff as attempt 1 (with its verify rank and the call number) and reset the tree for attempt 2.
/// Once per task; `None` when there is nothing to save or the reset fails (the tree is then left as it was).
pub fn checkpoint_attempt(env: &EnvPaths, rank: u8, call: usize) -> Option<String> {
    let a1 = env.agent_dir.join("attempt1.patch");
    if a1.exists() {
        return None;
    }
    let diff = ws_diff(env)?;
    std::fs::write(&a1, &diff).ok()?;
    if !git_apply(env, &a1, true) {
        let _ = std::fs::remove_file(&a1);
        return None;
    }
    let _ = std::fs::write(
        env.agent_dir.join("attempt1.meta"),
        format!("{rank} {call}"),
    );
    Some(format!(
        "\n🔁 ATTEMPT 1 SAVED (call {call}, not verified READY) and /workspace reset to the original code. Start attempt 2 now with a DIFFERENT root-cause hypothesis: view the code again, edit, verify. At the end the better-verified attempt is kept."
    ))
}

/// At the end of the budget: when attempt 1 ranks above the live tree, swap it back in. `current_rank` must reflect
/// a verify of the live tree made after the checkpoint (0 otherwise). Once per task.
pub fn select_attempt(env: &EnvPaths, current_rank: u8) -> Option<String> {
    let a1 = env.agent_dir.join("attempt1.patch");
    let done = env.agent_dir.join("attempt_selected");
    if !a1.exists() || done.exists() {
        return None;
    }
    let _ = std::fs::write(&done, "1");
    let meta = std::fs::read_to_string(env.agent_dir.join("attempt1.meta")).unwrap_or_default();
    let r1: u8 = meta
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if r1 <= current_rank {
        return None;
    }
    if let Some(cur) = ws_diff(env) {
        let a2 = env.agent_dir.join("attempt2.patch");
        std::fs::write(&a2, &cur).ok()?;
        if !git_apply(env, &a2, true) {
            return None;
        }
        if !git_apply(env, &a1, false) {
            let _ = git_apply(env, &a2, false);
            return None;
        }
    } else if !git_apply(env, &a1, false) {
        return None;
    }
    Some("\n🔁 ATTEMPT 1 KEPT: it verified better than attempt 2, so /workspace now holds attempt 1. Do not edit further.".to_string())
}

/// The answer to `/tmp/coder submit_patch` (or submit/finish/done/stop): how to finish the session.
pub fn submit_reply(subcmd: &str, has_submit: bool, last_verify_ready: bool) -> String {
    let ready = if last_verify_ready {
        "Your last verify said READY: finish now."
    } else {
        "Your last verify did not say READY: fix and verify first, or finish if you cannot."
    };
    if has_submit {
        format!(
            "`/tmp/coder` has no `{subcmd}` command: call the `submit_patch` tool itself (not through run_command). {ready}"
        )
    } else {
        format!(
            "There is no submit step: /workspace is graded exactly as it is now. {ready} To finish, reply with one sentence and NO tool call."
        )
    }
}

/// Flat call shape: engine hints are written for `run_skill_script` calls; when the model
/// reaches the engine through `run_command("/tmp/coder ...")` they must show that form instead, or the
/// model copies the nested JSON back into its calls.
pub fn flatten_hints(out: &str) -> String {
    let pair =
        Regex::new(r#"\\?"([A-Za-z_]+)\\?":\s*\\?"((?:[^"\\]|\\[^"])*)\\?""#).expect("regex");
    let args = Regex::new(r"`?args=\{([^{}]*)\}`?").expect("regex");
    let mut t = args
        .replace_all(out, |c: &regex::Captures| {
            let kv: Vec<String> = pair
                .captures_iter(&c[1])
                .map(|p| format!("{}='{}'", &p[1], &p[2]))
                .collect();
            if kv.is_empty() {
                String::new()
            } else {
                format!("`{}`", kv.join(" "))
            }
        })
        .into_owned();
    for (from, to) in [
        (
            r#"`run_skill_script` (`file_path="scripts/(\w+)\.py"`, `skill_name="swe-coder"`)"#,
            "`/tmp/coder $1`",
        ),
        (
            r#",?\s*`?file_path=\\?"scripts/(\w+)\.py\\?"`?"#,
            " `/tmp/coder $1`",
        ),
        (r#",?\s*`?skill_name=\\?"swe-coder\\?"`?"#, ""),
        (r"`?scripts/(\w+)\.py`?", "`/tmp/coder $1`"),
        (r"`?run_skill_script`?", "`/tmp/coder`"),
    ] {
        t = Regex::new(from)
            .expect("regex")
            .replace_all(&t, to)
            .into_owned();
    }
    // After a refused edit, restate the heredoc form: multi-line code quoted on the command line
    // is rejected by the shell.
    if t.contains("old_string")
        || t.contains("PATCH ERRORS")
        || t.contains("EMPTY DIFF")
        || t.contains("must specify")
    {
        t.push_str(
            "\nFLAT EDIT FORM: /tmp/coder patch file=pkg/mod.py symbol=Class.method <<'EOF' then a line <<<OLD, the exact current lines, a line ===, the new lines, a line >>>, then EOF. EOF appears exactly twice: at the end of the first line and alone on the last line; old and new are separated by ===, never by EOF (bash ends the text at the first EOF line and runs the rest as commands: `command not found`). Never quote multi-line code on the command line.",
        );
    }
    // The NEXT line after PATCH APPLIED must show the heredoc form of verify. It never carries a
    // verdict string: a `VERDICT: READY` inside a hint would be read as the verdict itself.
    if t.contains("PATCH APPLIED") {
        let next = Regex::new(r"(?m)^NEXT: .*verify.*$").expect("regex");
        t = next
            .replace(&t, "NEXT: your next call must be /tmp/coder verify <<'EOF' with a repro that asserts the behaviour the issue asks for, then EOF. This patch is NOT verified yet: submitting it now scores 0.")
            .into_owned();
    }
    t
}

/// Top-level CLI entrypoint discovering workspace/tmp paths automatically.
pub fn run_cli(argv: &[String], stdin_str: &str) -> Result<String> {
    let env = EnvPaths::discover();
    run_cli_with_env(&env, argv, stdin_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_dummy_git_repo() -> (TempDir, EnvPaths) {
        let tmp_root = TempDir::new().expect("tempdir");
        let ws = tmp_root.path().join("workspace");
        let tmp = tmp_root.path().join("tmp");
        std::fs::create_dir_all(ws.join("pkg")).unwrap();
        std::fs::create_dir_all(ws.join("tests")).unwrap();
        std::fs::create_dir_all(&tmp).unwrap();

        write_lf(
            &ws.join("pkg").join("__init__.py"),
            "from .validators import validate_username, AsyncValidator\n",
        )
        .unwrap();

        write_lf(
            &ws.join("pkg").join("validators.py"),
            r#"import re

def validate_username(name: str, allow_dots: bool = False) -> bool:
    """Validate a username string."""
    if not name:
        raise ValueError("username cannot be empty")
    pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, name))

def format_display_name(name: str) -> str:
    if not validate_username(name):
        raise ValueError("invalid username")
    return name.strip().lower()

class AsyncValidator:
    async def check(self, name: str, allow_dots: bool = False) -> bool:
        if not name:
            raise ValueError("username cannot be empty")
        pattern = r"^[a-zA-Z0-9_]+$"
        return bool(re.match(pattern, name))
"#,
        )
        .unwrap();

        write_lf(
            &ws.join("pkg").join("serializers.py"),
            r#"from .validators import validate_username

def serialize_user(user_dict: dict) -> dict:
    name = user_dict.get("username", "")
    if not validate_username(name):
        raise ValueError("bad user")
    return {"username": name}
"#,
        )
        .unwrap();

        write_lf(
            &ws.join("tests").join("test_validators.py"),
            r#"from pkg.validators import validate_username

def test_basic_username():
    assert validate_username("alice_1") is True
"#,
        )
        .unwrap();

        // Initialize git repo for `git diff` / `git merge-file` / `git status`
        Command::new("git")
            .args(["init"])
            .current_dir(&ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(&ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(&ws)
            .output()
            .unwrap();

        let env = EnvPaths::with_roots(ws, tmp);
        (tmp_root, env)
    }

    /// attempt 1 is saved and the tree reset; at the end it comes back only when it verified better.
    #[test]
    fn test_two_attempts_checkpoint_and_select() {
        let (_dir, env) = setup_dummy_git_repo();
        std::fs::create_dir_all(&env.agent_dir).unwrap();
        let f = env.ws.join("pkg").join("serializers.py");
        let orig = read_lf(&f).unwrap();
        write_lf(&f, &orig.replace("bad user", "attempt one")).unwrap();
        assert!(checkpoint_attempt(&env, 2, 14).is_some());
        assert_eq!(read_lf(&f).unwrap(), orig, "tree reset for attempt 2");
        assert!(checkpoint_attempt(&env, 2, 15).is_none(), "once per task");
        // attempt 2 verified worse (NEEDS FIX = 1): attempt 1 (rank 2) comes back
        write_lf(&f, &orig.replace("bad user", "attempt two")).unwrap();
        assert!(select_attempt(&env, 1).is_some());
        assert!(read_lf(&f).unwrap().contains("attempt one"));
        assert!(select_attempt(&env, 0).is_none(), "selected once");
        // a better attempt 2 is kept
        let (_d2, env2) = setup_dummy_git_repo();
        std::fs::create_dir_all(&env2.agent_dir).unwrap();
        let f2 = env2.ws.join("pkg").join("serializers.py");
        let o2 = read_lf(&f2).unwrap();
        write_lf(&f2, &o2.replace("bad user", "attempt one")).unwrap();
        assert!(checkpoint_attempt(&env2, 1, 14).is_some());
        write_lf(&f2, &o2.replace("bad user", "attempt two")).unwrap();
        assert!(select_attempt(&env2, 3).is_none());
        assert!(read_lf(&f2).unwrap().contains("attempt two"));
        assert_eq!(verdict_rank("x\nVERDICT: READY TO CALL"), 3);
        assert_eq!(verdict_rank("VERDICT: NEEDS FIX — ..."), 1);
    }

    /// a verify killed during its baseline tests left the files at HEAD; the next call restores the edit.
    #[test]
    fn test_interrupted_baseline_restores_patch() {
        let (_dir, env) = setup_dummy_git_repo();
        let f = env.ws.join("pkg").join("serializers.py");
        let edited = read_lf(&f).unwrap().replace("bad user", "invalid user");
        write_lf(&f, &edited).unwrap();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(&env.ws)
                .output()
                .unwrap()
        };
        std::fs::create_dir_all(&env.agent_dir).unwrap();
        let saved = env.agent_dir.join("baseline_stash.patch");
        std::fs::write(&saved, git(&["diff", "--binary", "HEAD"]).stdout).unwrap();
        git(&["checkout", "HEAD", "--", "pkg/serializers.py"]);
        assert!(!read_lf(&f).unwrap().contains("invalid user"));
        assert!(verify::restore_baseline_stash(&env));
        assert!(read_lf(&f).unwrap().contains("invalid user"));
        assert!(!saved.exists());
        assert!(!verify::restore_baseline_stash(&env));
    }

    /// a package whose import alone eats the repro limit gets a longer, measured limit once.
    #[test]
    fn test_slow_import_widens_repro_limit() {
        let (_dir, env) = setup_dummy_git_repo();
        std::fs::create_dir_all(env.ws.join("slowpkg")).unwrap();
        write_lf(
            &env.ws.join("slowpkg").join("__init__.py"),
            "import time\ntime.sleep(3)\nx = 1\n",
        )
        .unwrap();
        std::fs::create_dir_all(&env.agent_dir).unwrap();
        let code = "import os\nfrom slowpkg import x\nassert x\n";
        assert_eq!(verify::repro_limit(&env, 4), 4);
        let wider = verify::widen_repro_limit(&env, 4, code).expect("widened");
        assert!((16..=60).contains(&wider), "{wider}");
        assert_eq!(verify::repro_limit(&env, 4), wider);
        // measured once per session
        assert!(verify::widen_repro_limit(&env, 4, code).is_none());
        // a fast package keeps the limit
        let (_d2, env2) = setup_dummy_git_repo();
        std::fs::create_dir_all(&env2.agent_dir).unwrap();
        assert!(
            verify::widen_repro_limit(&env2, 15, "from pkg.validators import validate_username\n")
                .is_none()
        );
        assert_eq!(verify::repro_limit(&env2, 15), 15);
        // imports too slow for any limit with room for the repro: keep the short limit
        assert_eq!(verify::widened_limit(3, 4), Some(16));
        assert_eq!(verify::widened_limit(25, 15), Some(60));
        assert_eq!(verify::widened_limit(40, 15), Some(60));
        assert_eq!(verify::widened_limit(51, 15), None);
        assert_eq!(verify::widened_limit(5, 15), None);
    }

    /// a prompt reads EOF instead of waiting, and a timeout ends a grandchild that holds stdout.
    #[cfg(unix)]
    #[test]
    fn test_shell_timeout_kills_grandchildren_and_closes_stdin() {
        let dir = TempDir::new().expect("tmp");
        let (rc, out) = run_shell_with_timeout("read x; echo got:$x", dir.path(), dir.path(), 5);
        assert_eq!((rc, out.trim()), (0, "got:"));
        let t0 = Instant::now();
        let (rc, _) = run_shell_with_timeout("sleep 30; echo late", dir.path(), dir.path(), 1);
        assert_eq!(rc, 124);
        assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", t0.elapsed());
    }

    #[test]
    fn test_ast_graph_and_whitened_embeddings() {
        let (_dir, env) = setup_dummy_git_repo();
        let idx = load_or_build_index(&env).expect("build index");

        // Verify both sync `def` and `async def` are indexed
        assert!(idx.nodes.contains_key("pkg.validators.validate_username"));
        assert!(
            idx.nodes
                .contains_key("pkg.validators.AsyncValidator.check")
        );

        // Verify call graph edge: `format_display_name` calls `validate_username`
        let callers = idx
            .callers
            .get("pkg.validators.validate_username")
            .expect("callers");
        assert!(
            callers
                .iter()
                .any(|c| c.contains("format_display_name") || c.contains("serialize_user"))
        );

        // Verify `tests_of` maps `pkg.validators.validate_username` -> `tests/test_validators.py`
        let tests = idx
            .tests_of
            .get("pkg.validators.validate_username")
            .expect("tests_of");
        assert!(tests.contains(&"tests/test_validators.py".to_string()));

        // Verify 256-d whitened structural embedding finds `AsyncValidator.check` as structural sibling of `validate_username`
        let sib =
            crate::ast_graph::find_structural_sibling(&idx, "pkg.validators.validate_username");
        assert!(sib.is_some());
        let (sib_id, sim) = sib.unwrap();
        assert_eq!(sib_id, "pkg.validators.AsyncValidator.check");
        assert!(
            sim >= 0.55,
            "expected high structural cosine similarity, got {sim}"
        );
    }

    #[test]
    fn test_debug_localization_and_preloaded_code() {
        let (_dir, env) = setup_dummy_git_repo();
        let argv = vec![
            "debug".to_string(),
            "query=validate_username ignores allow_dots parameter in pkg/validators.py".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "").expect("debug");
        assert!(out.contains("TOP TARGETS (AST + Call/Inheritance Graph + BM25):"));
        assert!(out.contains("pkg/validators.py"));
        assert!(out.contains("validate_username"));
        assert!(out.contains("PRELOADED CODE"));
    }

    #[test]
    fn test_rename_sites_and_dunder_deletion() {
        let (_dir, env) = setup_dummy_git_repo();
        let idx = load_or_build_index(&env).expect("build index");
        let renames = crate::debug::find_rename_sites(
            &env.ws,
            &idx,
            "* `validators.validate_username` -> `.check_username`\n* rename format_display_name to display",
        );
        assert_eq!(renames.len(), 2, "{renames:?}");
        let (old, new, sites) = &renames[0];
        assert_eq!(
            (old.as_str(), new.as_str()),
            ("validate_username", "check_username")
        );
        // the def, its in-file caller and the serializers.py caller; never the test file
        assert_eq!(sites.len(), 3, "{sites:?}");
        assert!(sites.iter().any(|s| s.starts_with("pkg/serializers.py:5:")));
        assert!(!sites.iter().any(|s| s.starts_with("tests/")));

        let old_src = "class P:\n    def __init__(self):\n        self.a = 1\n\n    def go(self):\n        return self.a\n";
        let new_src = "class P:\n    def reset(self):\n        self.a = 1\n\n    def go(self):\n        return self.a\n";
        let msgs = crate::ast_graph::check_deleted_public_defs("pkg/p.py", old_src, new_src, "");
        assert!(msgs.iter().any(|m| m.contains("`P.__init__`")), "{msgs:?}");
        // a rename the issue asks for is not an accidental deletion; other deletions still are
        let old_go = "class P:\n    def start(self):\n        return 1\n\n    def stay(self):\n        return 2\n";
        let new_go = "class P:\n    def run(self):\n        return 1\n\n    def stay(self):\n        return 2\n";
        let issue = "* `P.start` -> `.run`";
        assert!(
            crate::ast_graph::check_deleted_public_defs("pkg/p.py", old_go, new_go, issue)
                .is_empty()
        );
        let new_both = "class P:\n    def run(self):\n        return 1\n";
        let msgs = crate::ast_graph::check_deleted_public_defs("pkg/p.py", old_go, new_both, issue);
        assert!(msgs.iter().any(|m| m.contains("`P.stay`")), "{msgs:?}");

        // in a dataclass too: `self.<module function>` is caught, fields are members
        let dc_old = "def lookup(x):\n    return x\n\n@dataclass\nclass T:\n    total: int = 0\n    hook: object = None\n\n    def get(self, k):\n        return k\n";
        let dc_new = dc_old.replace("return k", "self.hook(k)\n        return self.lookup(k)");
        let msgs =
            crate::ast_graph::check_new_undefined_and_self_methods("pkg/t.py", dc_old, &dc_new);
        assert!(msgs.iter().any(|m| m.contains("T.lookup")), "{msgs:?}");
        assert!(!msgs.iter().any(|m| m.contains("T.hook")), "{msgs:?}");
        let ga = dc_new.replace(
            "    total: int = 0\n",
            "    def __getattr__(self, n):\n        return n\n",
        );
        let ga_old = dc_old.replace(
            "    total: int = 0\n",
            "    def __getattr__(self, n):\n        return n\n",
        );
        assert!(
            crate::ast_graph::check_new_undefined_and_self_methods("pkg/t.py", &ga_old, &ga)
                .is_empty()
        );

        // dropping (or adding) @property breaks every existing access
        let p_old = "class C:\n    @property\n    def is_terminal(self) -> bool:\n        return True\n\n    @x.setter\n    def x(self, v):\n        pass\n";
        let p_new = p_old.replace("    @property\n", "");
        let msgs =
            crate::ast_graph::check_property_toggles("pkg/c.py", p_old, &p_new, "TTY env var");
        assert!(
            msgs.len() == 1 && msgs[0].contains("C.is_terminal"),
            "{msgs:?}"
        );
        assert!(
            crate::ast_graph::check_property_toggles("pkg/c.py", &p_new, p_old, "")[0]
                .contains("became")
        );
        assert!(crate::ast_graph::check_property_toggles("pkg/c.py", p_old, p_old, "").is_empty());
        let asked = "make `is_terminal` a method instead of a property";
        assert!(
            crate::ast_graph::check_property_toggles("pkg/c.py", p_old, &p_new, asked).is_empty()
        );

        let src = crate::verify::stub_runner_source(&["ruamel".into()], "/tmp/.agent/repro.py");
        assert!(
            src.contains("STUB = {\"ruamel\"}")
                && src.contains("run_path(\"/tmp/.agent/repro.py\"")
        );

        let swallow = "try:\n    f()\n    print('ok')\nexcept Exception as e:\n    print(e)\n";
        assert!(!crate::verify::vacuous_repro_note(swallow).is_empty());
        assert!(!crate::verify::vacuous_repro_note("print(f())\n").is_empty());
        assert!(crate::verify::vacuous_repro_note("from m import f\nassert f() == 2\n").is_empty());
        // "does not raise" is not a check: the only assert sits in `except`
        let no_raise =
            "from m import f\ntry:\n    f(1)\nexcept Exception as e:\n    assert False, e\n";
        assert!(!crate::verify::vacuous_repro_note(no_raise).is_empty());
        assert!(!crate::verify::vacuous_repro_note("f(1)\nassert False\n").is_empty());
        assert!(crate::verify::vacuous_repro_note("try:\n    f(1)\nexcept ValueError as e:\n    assert 'x' in str(e)\nassert f(2) == 3\n").is_empty());
        // a repro that imports only private names checks the edit, not the issue
        assert!(
            !crate::verify::private_only_repro_note(
                "from pkg.mod import _helper\nassert _helper(1) == 2\n"
            )
            .is_empty()
        );
        assert!(
            crate::verify::private_only_repro_note(
                "from pkg.mod import _helper, run\nassert run(1) == 2\n"
            )
            .is_empty()
        );
        assert!(
            crate::verify::private_only_repro_note("import pkg\nassert pkg.run(1) == 2\n")
                .is_empty()
        );

        let out = "_____ ERROR collecting tests/test_cli.py _____\nE   TypeError: metaclass conflict\n\
=========================== short test summary info ============================\n\
ERROR tests/test_cli.py - TypeError: metaclass conflict: the metaclass\n\
FAILED tests/test_a.py::test_x - AssertionError: 1 != 2\n!!! Interrupted: 1 error during collection !!!\n";
        let ids = crate::verify::failing_ids(out);
        assert_eq!(
            ids.into_iter().collect::<Vec<_>>(),
            vec![
                "tests/test_a.py::test_x".to_string(),
                "tests/test_cli.py".to_string()
            ]
        );
    }

    #[test]
    fn test_debug_keeps_issue_text_for_later_checks() {
        let (_dir, env) = setup_dummy_git_repo();
        let argv = vec![
            "debug".to_string(),
            "issue=* `validators.validate_username` -> `.check_username`".to_string(),
            "query=validate_username".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "").expect("debug");
        // the rename is only visible in the issue text, not in the query
        assert!(
            out.contains("RENAME `validate_username` -> `check_username`"),
            "{out}"
        );
        let saved = read_lf(&env.agent_dir.join("issue.txt")).unwrap();
        assert!(saved.contains("check_username") && saved.contains("validate_username"));

        // a backticked identifier in the issue ranks its definition even if the query
        // paraphrases it
        let argv = vec![
            "debug".to_string(),
            "issue=New users fail: `serialize_user` must keep the name.".to_string(),
            "query=user dict conversion".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "").expect("debug");
        let top = out.split("TOP FILES").next().unwrap_or("");
        assert!(top.contains("serialize_user"), "{out}");
    }

    #[test]
    fn test_view_misnamed_method_is_not_called_inherited() {
        let (_dir, env) = setup_dummy_git_repo();
        let argv = vec![
            "view".to_string(),
            "file=pkg/validators.py".to_string(),
            "symbol=AsyncValidator.chek".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "").expect("view");
        assert!(out.contains("not defined anywhere"), "{out}");
        assert!(
            out.contains("Did you mean `AsyncValidator.check`?"),
            "{out}"
        );
        assert!(!out.contains("inherits"), "{out}");
    }

    #[test]
    fn test_patch_refuses_skill_script_paths() {
        let (_dir, env) = setup_dummy_git_repo();
        let argv = vec![
            "patch".to_string(),
            "file=scripts/verify.py".to_string(),
            "create=1".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "import x\n").expect("patch");
        assert!(out.contains("saved as the repro"), "{out}");
        assert!(!env.ws.join("scripts").join("verify.py").exists());
        let saved = read_lf(&env.agent_dir.join("repro.py")).unwrap();
        assert_eq!(saved, "import x\n");
        let argv = vec![
            "patch".to_string(),
            "file=scripts/view.py".to_string(),
            "create=1".to_string(),
        ];
        let out = run_cli_with_env(&env, &argv, "import x\n").expect("patch");
        assert!(out.contains("refused to edit `scripts/view.py`"), "{out}");
        assert!(crate::patch::is_skill_script_path("./scripts/patch.py"));
        assert!(!crate::patch::is_skill_script_path("pkg/scripts/patch.py"));
    }

    #[test]
    fn test_fuzzy_realign_one_char_slip() {
        let live = "class R:\n    def include_router(\n        self,\n        router,\n        *,\n        prefix: str = \"\",\n    ):\n        pass\n";
        // `*` for `*,` in both old and new
        let old = "        router,\n        *\n        prefix: str = \"\",";
        let new =
            "        router,\n        *\n        prefix: str = \"\",\n        check: bool = False,";
        let (fo, fnew, note) =
            crate::patch::fuzzy_realign(live, old, new, Some((2, 8))).expect("one window");
        assert_eq!(
            fo,
            "        router,\n        *,\n        prefix: str = \"\","
        );
        assert!(
            fnew.contains("        *,\n") && fnew.ends_with("check: bool = False,"),
            "{fnew}"
        );
        assert!(note.contains("`*`"), "{note}");
        // an edit resent after it applied is recognised; single lines are too weak a signal
        let src = "def f():\n    a = 1\n    b = 2\n    return a\n";
        assert!(crate::patch::already_applied(src, "  a = 1\n\n  b = 2"));
        assert!(!crate::patch::already_applied(src, "    a = 1"));
        assert!(!crate::patch::already_applied(src, "a = 1\nb = 3"));
        // pytest's budget shrinks with the session: 65 s early, 40 % of the rest later, 15 s floor
        assert_eq!(crate::verify::pytest_budget_secs(0, 1.0), 65);
        assert_eq!(crate::verify::pytest_budget_secs(160, 1.0), 40);
        assert_eq!(crate::verify::pytest_budget_secs(250, 1.0), 15);
        assert_eq!(crate::verify::pytest_budget_secs(400, 1.0), 15);
        assert_eq!(crate::verify::pytest_budget_secs(160, 2.0), 65);
        // ... unless the one new line is an edit of the one old line, which is gone
        let src =
            "    if i == n:\n        column = Column(_index=index, highlight=self.highlight)\n";
        let (old, new) = (
            "        column = Column(_index=index)",
            "        column = Column(_index=index, highlight=self.highlight)",
        );
        assert!(crate::patch::already_applied_line(src, old, new));
        // old still present, unrelated old line, or a short new line: not applied
        assert!(!crate::patch::already_applied_line(
            &format!("{src}{old}\n"),
            old,
            new
        ));
        assert!(!crate::patch::already_applied_line(
            src,
            "    x = compute()",
            new
        ));
        assert!(!crate::patch::already_applied_line(
            "    return x\n",
            "    return y",
            "    return x"
        ));
        // a miss deep inside a long function shows that region, not the signature
        let mut long = String::from("def f(a):\n");
        for i in 0..60 {
            long.push_str(&format!("    x{i} = a + {i}\n"));
        }
        let (s, e) = crate::patch::closest_window(&long, "    x47 = a + 470\n", 1, 61, 25);
        assert!(s <= 48 && 48 <= e && s > 30, "{s}..{e}");
        // no window when two lines are far off, or when everything differs
        assert!(crate::patch::fuzzy_realign(live, "    x = 1\n    y = 2", "z", None).is_none());

        // trailing comments copied from an annotated issue snippet do not block the match
        let src = "def f(a):\n    if a:  # yes\n        b = '#x'\n    else:\n        b = 2\n    return b\n";
        let old =
            "    if a:\n        b = '#x'  # condition1\n    else: # condition2\n        b = 2";
        let (o, _, note) = crate::patch::fuzzy_realign(src, old, "    pass", None).expect("window");
        assert!(o.contains("if a:  # yes") && o.contains("else:\n"), "{o}");
        assert!(note.contains("3 line(s)"), "{note}");
        // a docstring retyped with `"""` where the file has `"` still matches
        let src = "    def du(self, *o):\n        \"du(*o) -> discard\"\n        for v in o:\n            self.discard(v)\n";
        let old =
            "    def du(self, *o):\n        \"\"\"du(*o) -> discard\"\"\"\n        for v in o:";
        let new =
            "    def du(self, *o):\n        \"\"\"du(*o) -> discard\"\"\"\n        self.bulk(o)";
        let (o, n, _) = crate::patch::fuzzy_realign(src, old, new, None).expect("quote window");
        assert!(o.contains("\"du(*o) -> discard\"\n"), "{o}");
        assert!(n.contains("        \"du(*o) -> discard\"\n"), "{n}");
        assert_eq!(
            crate::patch::strip_trailing_comment("s = 'a#b'  # c"),
            "s = 'a#b'"
        );
        // `from src.pkg import ...` in a src/ layout gets a hint
        let ws = std::env::temp_dir().join(format!("swe_srcnote_{}", std::process::id()));
        std::fs::create_dir_all(ws.join("src").join("jh")).unwrap();
        let note = crate::verify::src_import_note("import os\nfrom src.jh.parser import P\n", &ws);
        assert!(note.contains("`jh`"), "{note}");
        assert!(crate::verify::src_import_note("from jh import P\n", &ws).is_empty());
        // ... and is rewritten to the installed package before the repro runs
        let (fixed, pkg) = crate::verify::rewrite_src_imports(
            "from src.jh.parser import P\nx = src.jh.parser.f()\nimport srcs\n",
            &ws,
        )
        .expect("rewrite");
        assert_eq!(pkg, "jh");
        assert_eq!(
            fixed,
            "from jh.parser import P\nx = jh.parser.f()\nimport srcs\n"
        );
        assert!(crate::verify::rewrite_src_imports("from jh import P\n", &ws).is_none());
        assert!(crate::verify::rewrite_src_imports("from src.other import P\n", &ws).is_none());
        // flat call shape: argv exactly as `sh -c` splits the commands the prompt teaches
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let (cmd, a, _) =
            parse_cli_and_stdin(&v(&["debug", "query=parse_args `Foo.bar` KeyError"]), "");
        assert_eq!(
            (cmd.as_str(), a["query"].as_str()),
            ("debug", "parse_args `Foo.bar` KeyError")
        );
        let (_, a, _) = parse_cli_and_stdin(
            &v(&["debug", "query=parse_args", "Foo.bar", "KeyError"]),
            "",
        );
        assert_eq!(a["query"], "parse_args Foo.bar KeyError");
        let (cmd, a, _) = parse_cli_and_stdin(&v(&["view", "file=pkg/mod.py", "symbol=Cls.m"]), "");
        assert_eq!(
            (cmd.as_str(), a["file"].as_str(), a["symbol"].as_str()),
            ("view", "pkg/mod.py", "Cls.m")
        );
        // `new=` becomes the patch body (the replacement text), `old=` stays an argument
        let (_, a, new_body) = parse_cli_and_stdin(
            &v(&[
                "patch",
                "file=pkg/mod.py",
                "symbol=f",
                r"old=    x = 1\n    return x",
                r"new=    x = 2\n    return x",
            ]),
            "",
        );
        assert_eq!(
            (a["old"].as_str(), new_body.as_str()),
            ("    x = 1\n    return x", "    x = 2\n    return x")
        );
        // a `\'` written inside shell single quotes is a plain quote in code
        let (_, a, _) = parse_cli_and_stdin(
            &v(&[
                "patch",
                "file=m.py",
                r"old=if s.startswith(\'x\'):",
                "new=pass",
            ]),
            "",
        );
        assert_eq!(a["old"], "if s.startswith('x'):");
        let (cmd, a, body) =
            parse_cli_and_stdin(&v(&["verify"]), "from m import f\nassert f() == 2\n");
        assert_eq!(cmd, "verify");
        assert!(a.is_empty() && body.contains("assert f() == 2"), "{body}");
        let (_, a, _) = parse_cli_and_stdin(
            &v(&["verify", r"repro=from m import f\nassert f() == 2"]),
            "",
        );
        assert_eq!(a["repro"], "from m import f\nassert f() == 2");
        // a real newline is never re-decoded (a repro that prints "\n" stays intact)
        let (_, a, _) = parse_cli_and_stdin(&v(&["verify", "repro=print('a\\nb')\nassert 1"]), "");
        assert!(a["repro"].contains("'a\\nb'"), "{}", a["repro"]);

        // flat mode rewrites the engine's skill-call hints (no nested JSON left for the model to copy)
        for hint in [
            "NEXT: Call `run_skill_script` with `args={\"repro\": \"import ...\\nassert ...\"}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"` to verify.",
            "revise via `run_skill_script` (`file_path=\"scripts/patch.py\"`, `skill_name=\"swe-coder\"`) or view the code",
            "Run `scripts/debug.py` (`query=\"...\"`) if you need it; then `scripts/view.py`.",
        ] {
            let f = flatten_hints(hint);
            assert!(
                !f.contains("run_skill_script")
                    && !f.contains("scripts/")
                    && !f.contains("skill_name"),
                "{f}"
            );
            assert!(f.contains("/tmp/coder"), "{f}");
        }
        assert!(
            flatten_hints("NEXT: `args={\"file\": \"a.py\", \"symbol\": \"f\"}`")
                .contains("file='a.py' symbol='f'")
        );
        assert!(flatten_hints("VERDICT: EMPTY DIFF").contains("FLAT EDIT FORM"));
        // `/tmp/coder submit_patch` explains how to finish
        let r = submit_reply("submit_patch", false, true);
        assert!(
            r.contains("no submit step") && r.contains("READY") && r.contains("NO tool call"),
            "{r}"
        );
        assert!(submit_reply("submit", true, false).contains("`submit_patch` tool itself"));
        // the no-edit checkpoint's search hint becomes a flat debug call
        let f = flatten_hints(
            "search the whole repo once with `scripts/debug.py` (`args={\"query\": \"<exact identifier>\"}`); otherwise",
        );
        assert!(
            f.contains("/tmp/coder debug") && f.contains("query='<exact identifier>'"),
            "{f}"
        );
        // after an applied patch the next step is a heredoc verify, not submit
        let f = flatten_hints(
            "PATCH APPLIED (1 block(s))\nNEXT: Call `run_skill_script` with `args={\"repro\": \"import ...\"}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"` to verify and unlock `submit_patch`.\n",
        );
        assert!(
            f.contains("/tmp/coder verify <<'EOF'")
                && f.contains("NOT verified yet")
                && !f.contains("VERDICT"),
            "{f}"
        );
        assert!(!f.contains("repro='"), "{f}");
        // a unified-diff hunk in a patch body becomes old/new text; plain code does not
        let (o, n) = crate::patch::unified_hunk_to_old_new(
            "@@ -1,3 +1,3 @@\n     if obj is None:\n-        return self\n+        return self.wrap(obj)\n",
        )
        .expect("hunk");
        assert_eq!(o, "    if obj is None:\n        return self");
        assert_eq!(n, "    if obj is None:\n        return self.wrap(obj)");
        assert!(crate::patch::unified_hunk_to_old_new("def f():\n    return 1\n").is_none());
        // `old / === / new` without <<<OLD / >>> markers
        let (o, n) =
            crate::patch::bare_separator_old_new("    a = 1\n===\n    a = 2\n").expect("bare");
        assert_eq!((o.as_str(), n.as_str()), ("    a = 1", "    a = 2"));
        assert!(crate::patch::bare_separator_old_new("a\n===\nb\n===\nc").is_none());
        assert!(crate::patch::bare_separator_old_new("===\nb").is_none());
        // the same error signature with different arguments
        assert_eq!(
            error_signature("PATCH ERRORS:\n  ! file `README.md` does not exist in `/workspace`"),
            error_signature(
                "PATCH ERRORS:\n  ! file `docs/README.rst` does not exist in `/workspace`"
            )
        );
        assert!(error_signature("PATCH APPLIED (1 block(s))").is_none());
        // `--- path:A-B` + old / === / new / >>> = the separator form, header dropped
        assert!(
            crate::patch::header_range_body("--- p.py:906-909\n    a = 1\n===\n    a = 2\n>>>\n")
                .is_none()
        );
        let mut pp = BTreeMap::new();
        pp.insert("file".to_string(), "p.py".to_string());
        let blocks = crate::patch::parse_patch_blocks(
            &pp,
            "--- p.py:906-909\n    a = 1\n===\n    a = 2\n>>>\n",
        );
        assert_eq!(blocks.len(), 1);
        let (old_clean, _) = crate::patch::strip_gutter(blocks[0].old.as_deref().unwrap_or(""));
        assert_eq!(old_clean, "    a = 1");
        assert_eq!(blocks[0].new_text, "    a = 2");
        // a unified diff closed with a `>>>` marker: the marker is dropped
        let blocks = crate::patch::try_parse_unified_diff(
            "ReadMe.md",
            "--- ReadMe.md\n+++ ReadMe.md\n@@ -1,2 +1,3 @@\n ## Quickstart\n-old line\n+new line\n+another\n>>>\n",
        )
        .expect("diff");
        assert_eq!(blocks[0].new_text, "## Quickstart\nnew line\nanother");
        // stale old text from the original file: only the changed lines apply onto the current file
        let live = "def f(x):\n    y = clean(x)\n    if y:\n        return 1\n    return 0\n";
        let old = "def f(x):\n    if x:\n        return 1\n    return 0";
        let new = "def f(x):\n    if x:\n        return 2\n    return 0";
        assert_eq!(
            crate::patch::rebase_stale_edit(live, old, new).as_deref(),
            Some("def f(x):\n    y = clean(x)\n    if y:\n        return 2\n    return 0\n")
        );
        // ambiguous changed line: refused
        assert!(
            crate::patch::rebase_stale_edit("a = 1\nb = 1\na = 1\n", "a = 1", "a = 2").is_none()
        );
        // an insertion anchor must be a whole line, never a prefix of a longer one
        assert!(
            crate::patch::rebase_stale_edit(
                "from os import PathLike, stat, BrokenPipe\n",
                "from os import PathLike, stat",
                "from os import PathLike, stat\nimport errno"
            )
            .is_none()
        );
        // old text with two adjacent raw strings merged and backslashes doubled
        let live = "def g():\n    v = [\n        (\n            r\"(?:[A-Z0-9]\\.)+\"\n            r\"(?:[A-Z]{2,6}\\.?)\"\n        ),\n        r\"localhost\",\n    ]\n";
        let old = "        (\n            r\"(?:[A-Z0-9]\\\\.)+(?:[A-Z]{2,6}\\\\.?)\"\n        ),";
        let new = "        (\n            r\"(?:[\\w]\\\\.)+(?:[\\w]{2,6}\\\\.?)\"\n        ),";
        assert_eq!(
            crate::patch::loose_realign(live, old, new).as_deref(),
            Some(
                "def g():\n    v = [\n        (\n            r\"(?:[\\w]\\.)+(?:[\\w]{2,6}\\.?)\"\n        ),\n        r\"localhost\",\n    ]\n"
            )
        );
        // two equal windows: refused
        assert!(
            crate::patch::loose_realign(
                "total = compute(a, b)\ny = 2\ntotal = compute(a,b)\n",
                "total = compute(a,  b)",
                "total = 3"
            )
            .is_none()
        );
        // old / === / new / === : the second separator closes the block
        let (o, n) =
            crate::patch::bare_separator_old_new("    a = 1\n===\n    a = 2\n    b = 3\n===\n")
                .expect("closing ===");
        assert_eq!(
            (o.as_str(), n.as_str()),
            ("    a = 1", "    a = 2\n    b = 3")
        );
        // an abbreviated docstring kept the same in old and new: the edit applies around the file's docstring
        let live = "class A:\n    def floor(self, f):\n        \"\"\"Long doc.\n\n        >>> A().floor('h')\n        \"\"\"\n\n        return self.span(f)[0]\n";
        let old = "    def floor(self, f):\n        \"\"\"Long doc.\"\"\"\n\n        return self.span(f)[0]";
        let new = "    def floor(self, f, **kw):\n        \"\"\"Long doc.\"\"\"\n\n        return self.span(f, **kw)[0]";
        let out = crate::patch::edit_around_docstring(live, old, new).expect("split edit");
        assert!(
            out.contains("def floor(self, f, **kw):")
                && out.contains("return self.span(f, **kw)[0]"),
            "{out}"
        );
        assert!(
            out.contains(">>> A().floor('h')"),
            "the file's docstring must stay: {out}"
        );
        assert!(
            crate::patch::edit_around_docstring(
                live,
                old,
                "    def floor(self, f):\n        \"\"\"Other.\"\"\"\n"
            )
            .is_none()
        );
        // leaked template tokens end the body; a blank line missing from old_string still matches
        assert_eq!(
            crate::patch::cut_template_leak(
                "-    a = 1\n+    a = 2\n```}<tool_call|>\n<|channel>thought\nmore"
            ),
            "-    a = 1\n+    a = 2\n"
        );
        assert_eq!(crate::patch::cut_template_leak("x = 1\n"), "x = 1\n");
        let src = "    \"\"\"doc\"\"\"\n\n    parent = [1]\n    if not parent:\n        return\n";
        assert_eq!(
            crate::patch::blankless_realign(
                src,
                "    \"\"\"doc\"\"\"\n    parent = [1]\n    if not parent:"
            )
            .as_deref(),
            Some("    \"\"\"doc\"\"\"\n\n    parent = [1]\n    if not parent:")
        );
        assert!(crate::patch::blankless_realign(src, "    nothing = 0\n    here = 1").is_none());
        // root-level scratch files leave the workspace; package files and real sources stay
        let sw = TempDir::new().unwrap();
        let st = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(sw.path())
            .status()
            .unwrap();
        std::fs::create_dir_all(sw.path().join("pkg")).unwrap();
        for f in [
            "repro.py",
            "out.txt",
            "a.json",
            "file1.txt",
            "pkg/repro.py",
            "setup_helper.py",
        ] {
            std::fs::write(sw.path().join(f), "x").unwrap();
        }
        let mut moved = crate::verify::move_root_scratch(sw.path(), st.path());
        moved.sort();
        assert_eq!(moved, ["a.json", "file1.txt", "out.txt", "repro.py"]);
        assert!(
            sw.path().join("pkg/repro.py").exists() && sw.path().join("setup_helper.py").exists()
        );
        // `README.md` finds `ReadMe.md` when it is the only case-insensitive match
        let ci = TempDir::new().unwrap();
        std::fs::create_dir_all(ci.path().join("Docs")).unwrap();
        std::fs::write(ci.path().join("ReadMe.md"), "x").unwrap();
        std::fs::write(ci.path().join("Docs/Guide.md"), "x").unwrap();
        assert_eq!(
            case_insensitive_path(ci.path(), "README.md").as_deref(),
            Some("ReadMe.md")
        );
        assert_eq!(
            case_insensitive_path(ci.path(), "docs/guide.md").as_deref(),
            Some("Docs/Guide.md")
        );
        assert!(case_insensitive_path(ci.path(), "missing.md").is_none());
        // a unified diff keeps the call's symbol
        let mut sa = BTreeMap::new();
        sa.insert("file".to_string(), "typer/core.py".to_string());
        sa.insert("symbol".to_string(), "TyperOption.__init__".to_string());
        let blocks = crate::patch::parse_patch_blocks(
            &sa,
            "--- typer/core.py\n+++ typer/core.py\n@@ -436,2 +436,2 @@\n         x: int = 1,\n-        y = 2\n+        y = 3\n",
        );
        assert_eq!(blocks[0].symbol.as_deref(), Some("TyperOption.__init__"));
        // git-conflict markers with a 7-char `=======` are a separator, not a batch header
        let mut pa = BTreeMap::new();
        pa.insert("file".to_string(), "g/l.py".to_string());
        let blocks = crate::patch::parse_patch_blocks(
            &pa,
            "<<<<<<< OLD\n    a = 1\n=======\n    a = 2\n>>>>>>>\n",
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file, "g/l.py");
        assert_eq!(blocks[0].old.as_deref(), Some("    a = 1"));
        // `--- path:A-B` headers + replacement lines only = replace lines A..B
        let (p, a, b, n) = crate::patch::header_range_body(
            "--- g/x.py:104-110\n+++ g/x.py:104-110\n            if m:\n                pass\n",
        )
        .expect("range body");
        assert_eq!((p.as_str(), a, b), ("g/x.py", 104, 110));
        assert_eq!(n, "            if m:\n                pass");
        assert!(
            crate::patch::header_range_body(
                "--- g/x.py:1-3\n+++ g/x.py:1-3\n@@ -1,3 +1,3 @@\n-a\n+b\n"
            )
            .is_none()
        );
        // one marker only: `old / === / new / >>>`
        let (o, n) =
            crate::patch::bare_separator_old_new("    a = 1\n===\n    a = 2\n>>>\n").expect("half");
        assert_eq!((o.as_str(), n.as_str()), ("    a = 1", "    a = 2"));
        // diff headers with a glued line number and an indented closing EOF
        let body = crate::patch::strip_stray_eof(
            "--- src/m.py:621\n+++ src/m.py:621\n@@ -621,2 +621,2 @@\n     a = 1\n-    b = 2\n+    b = 3\n EOF\n",
        );
        let blocks = crate::patch::try_parse_unified_diff("src/m.py", &body).expect("diff");
        assert_eq!(blocks[0].file, "src/m.py");
        assert_eq!(blocks[0].old.as_deref(), Some("    a = 1\n    b = 2"));
        // the view header copied as the first old line is dropped
        let (o, _) = crate::patch::strip_gutter(
            "--- typer/core.py:677-694 (function T.r) ---\n    def r(self):\n        return 1",
        );
        assert_eq!(o, "    def r(self):\n        return 1");
        assert_eq!(crate::patch::strip_stray_eof("x = 1\n"), "x = 1\n");
        // a line range plus a note, as in our own `view` header
        let blocks = crate::patch::try_parse_unified_diff(
            "x.py",
            "--- src/m.py:676-693 (function F.g, file has 1209 lines)\n+++ src/m.py:676-693 (function F.g)\n@@ -676,1 +676,1 @@\n-    b = 2\n+    b = 3\n",
        )
        .expect("diff");
        assert_eq!(blocks[0].file, "src/m.py");
        assert!(
            crate::patch::bare_separator_old_new("    \"\"\"Doc\n    ===\n    \"\"\"").is_none()
        );
        assert!(crate::patch::unified_hunk_to_old_new("    x = 1\n    y = 2\n").is_none());

        // legacy twins rank below their current form
        let files: std::collections::HashSet<String> = [
            "fastapi/_compat/v1.py",
            "fastapi/_compat/v2.py",
            "pkg/legacy/io.py",
            "pkg/io.py",
            "pkg/only/v1.py",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert!(crate::debug::is_legacy_twin(
            "fastapi/_compat/v1.py",
            &files
        ));
        assert!(!crate::debug::is_legacy_twin(
            "fastapi/_compat/v2.py",
            &files
        ));
        assert!(crate::debug::is_legacy_twin("pkg/legacy/io.py", &files));
        assert!(!crate::debug::is_legacy_twin("pkg/only/v1.py", &files));
        let _ = std::fs::remove_dir_all(&ws);

        // the line range offered after a miss covers the lines the model was aiming at
        let src = "def g():\n    a = 1\n    b = compute(a)\n    c = b + 1\n    return c\n";
        let old = "    b = compute(a, x)\n    c = b + 1\n    return c";
        assert_eq!(crate::patch::best_line_window(src, old, 1, 5), Some((3, 5)));
        assert_eq!(crate::patch::best_line_window(src, "zzz\nqqq", 1, 5), None);
    }

    #[test]
    fn test_view_sync_async_and_multi_target() {
        let (_dir, env) = setup_dummy_git_repo();

        // 1. Bare `symbol=AsyncValidator.check` without `file=`
        let out_bare = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "symbol=AsyncValidator.check".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(out_bare.contains("async def check"));

        // 2. Multi-target `targets=...`
        let out_multi = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "targets=pkg/validators.py:validate_username,pkg/serializers.py:serialize_user"
                    .to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(out_multi.contains("def validate_username"));
        assert!(out_multi.contains("def serialize_user"));
    }

    #[test]
    fn test_patch_multi_file_and_3way_live_merge() {
        let (_dir, env) = setup_dummy_git_repo();

        // Step 1: Patch `validate_username` (expanding it by +4 lines) AND `pkg/serializers.py` in a single batch call
        let batch_body = r#"=== file=pkg/validators.py symbol=validate_username ===
def validate_username(name: str, allow_dots: bool = False) -> bool:
    """Validate a username string."""
    if not name:
        raise ValueError("username cannot be empty")
    if allow_dots:
        pattern = r"^[a-zA-Z0-9_.]+$"
    else:
        pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, name))
=== file=pkg/serializers.py old="if not validate_username(name):" ===
if not validate_username(name, allow_dots=True):
"#;
        let out1 = run_cli_with_env(&env, &["patch".to_string()], batch_body).unwrap();
        assert!(
            out1.contains("PATCH APPLIED (2 block(s)"),
            "out1 was: {out1}"
        );
        assert!(out1.contains("STATIC CRITIC: CLEAN"), "out1 was: {out1}");

        // Step 2: Patch `AsyncValidator.check` in `pkg/validators.py` (same file!) and verify both edits coexist cleanly
        let second_body = r#"async def check(self, name: str, allow_dots: bool = False) -> bool:
    if not name:
        raise ValueError("username cannot be empty")
    pattern = r"^[a-zA-Z0-9_.]+$" if allow_dots else r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, name))
"#;
        let out2 = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=AsyncValidator.check".to_string(),
            ],
            second_body,
        )
        .unwrap();
        assert!(out2.contains("PATCH APPLIED"), "out2 was: {out2}");

        let final_validators = read_lf(&ws_file(&env.ws, "pkg", "validators.py")).unwrap();
        assert!(final_validators.contains("if allow_dots:\n        pattern ="));
        assert!(final_validators.contains("pattern = r\"^[a-zA-Z0-9_.]+$\" if allow_dots"));

        // Verify `/tmp/.agent/best.patch` was saved
        assert!(env.agent_dir.join("best.patch").is_file());
    }

    fn ws_file(ws: &Path, dir: &str, name: &str) -> PathBuf {
        ws.join(dir).join(name)
    }

    #[test]
    fn test_verify_hygiene_and_best_patch_restore() {
        let (_dir, env) = setup_dummy_git_repo();

        // Apply a valid patch first
        let body = r#"def format_display_name(name: str) -> str:
    if not validate_username(name, allow_dots=True):
        raise ValueError("invalid username")
    return name.strip().lower()
"#;
        let _ = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=format_display_name".to_string(),
            ],
            body,
        )
        .unwrap();

        // Simulate accidental workspace wipe + stray scratch file + test edit
        Command::new("git")
            .args(["checkout", "HEAD", "--", "pkg/validators.py"])
            .current_dir(&env.ws)
            .output()
            .unwrap();
        write_lf(&env.ws.join("scratch_debug.py"), "print(1)\n").unwrap();
        write_lf(
            &ws_file(&env.ws, "tests", "test_validators.py"),
            "# accidental test edit\n",
        )
        .unwrap();

        let verify_out = run_cli_with_env(&env, &["verify".to_string()], "").unwrap();
        assert!(
            verify_out.contains("Restored `/tmp/.agent/best.patch`"),
            "verify_out: {verify_out}"
        );
        assert!(
            !env.ws.join("scratch_debug.py").exists(),
            "untracked scratch file should be removed"
        );
        let test_txt = read_lf(&ws_file(&env.ws, "tests", "test_validators.py")).unwrap();
        assert!(
            test_txt.contains("def test_basic_username"),
            "modified test file should be reverted to HEAD"
        );
        assert!(
            verify_out.contains("VERDICT: READY TO CALL `submit_patch` NOW."),
            "verify_out: {verify_out}"
        );
    }

    #[test]
    fn test_class_method_upsert_and_nested_function_upsert() {
        let (_dir, env) = setup_dummy_git_repo();

        // Add a second method and an outer function with a nested helper
        let val_path = ws_file(&env.ws, "pkg", "validators.py");
        let mut orig = read_lf(&val_path).unwrap();
        orig.push_str(
            r#"    def version(self) -> int:
        return 1

def build_table(rows: list) -> list:
    def add_cell(item: str) -> str:
        return item.strip()
    out = []
    for r in rows:
        out.append(add_cell(r))
    return out
"#,
        );
        write_lf(&val_path, &orig).unwrap();

        // 1. Pass `symbol=AsyncValidator` (the class!) with only `async def check(...)` -> must replace ONLY `AsyncValidator.check` and keep `version`
        let method_body = r#"async def check(self, name: str, allow_dots: bool = False) -> bool:
    return bool(name)
"#;
        let out_cls_replace = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=AsyncValidator".to_string(),
            ],
            method_body,
        )
        .unwrap();
        assert!(
            out_cls_replace.contains("replaced symbol `AsyncValidator.check`"),
            "got: {out_cls_replace}"
        );
        let after1 = read_lf(&val_path).unwrap();
        assert!(
            after1.contains(
                "    async def check(self, name: str, allow_dots: bool = False) -> bool:"
            ),
            "class method `check` must stay indented at 4 spaces inside `AsyncValidator`! got:\n{after1}"
        );
        assert!(
            after1.contains("def version(self) -> int:"),
            "class method `version` must be preserved!"
        );

        // 2. Pass `symbol=AsyncValidator` with a brand-new method `def is_ready(self) -> bool:` -> must insert into `AsyncValidator`
        let new_method_body = r#"def is_ready(self) -> bool:
    return True
"#;
        let out_cls_insert = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=AsyncValidator".to_string(),
            ],
            new_method_body,
        )
        .unwrap();
        assert!(
            out_cls_insert.contains("inserted new method `AsyncValidator.is_ready`"),
            "got: {out_cls_insert}"
        );
        let after2 = read_lf(&val_path).unwrap();
        assert!(after2.contains("def version(self) -> int:"));
        assert!(after2.contains("    def is_ready(self) -> bool:"));

        // 3. Pass `symbol=build_table` (outer function) with only `def add_cell(...)`
        // -> must replace ONLY nested `add_cell` and preserve `for r in rows:` in `build_table`
        let nested_body = r#"def add_cell(item: str) -> str:
    return item.strip().upper()
"#;
        let out_nested = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=build_table".to_string(),
            ],
            nested_body,
        )
        .unwrap();
        assert!(
            out_nested.contains("replaced symbol `add_cell`"),
            "got: {out_nested}"
        );
        let after3 = read_lf(&val_path).unwrap();
        assert!(after3.contains("def build_table(rows: list) -> list:"));
        assert!(after3.contains("return item.strip().upper()"));
        assert!(after3.contains("for r in rows:"));
    }

    #[test]
    fn test_auto_file_resolution_and_no_accidental_file_creation() {
        let (_dir, env) = setup_dummy_git_repo();

        // Run debug first so `last_targets.json` is populated
        let _ = run_cli_with_env(
            &env,
            &[
                "debug".to_string(),
                "query=validate_username in validators.py".to_string(),
            ],
            "",
        )
        .unwrap();

        // 1. Omit `file=` completely with `<<<OLD ... === ... >>>` -> auto-discovers `pkg/validators.py`
        let search_replace = r#"<<<OLD
    pattern = r"^[a-zA-Z0-9_]+$"
===
    pattern = r"^[a-zA-Z0-9_.]+$" if allow_dots else r"^[a-zA-Z0-9_]+$"
>>>"#;
        let out_auto = run_cli_with_env(&env, &["patch".to_string()], search_replace).unwrap();
        assert!(
            out_auto.contains("PATCH APPLIED") && out_auto.contains("pkg/validators.py"),
            "got: {out_auto}"
        );

        // 2. Pass suffix `file=serializers.py` and `snippet=` / `replacement=` aliases -> resolves to `pkg/serializers.py`
        let out_alias = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=serializers.py".to_string(),
                "snippet=if not validate_username(name):".to_string(),
                "replacement=if not validate_username(name, allow_dots=True):".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            out_alias.contains("PATCH APPLIED") && out_alias.contains("pkg/serializers.py"),
            "got: {out_alias}"
        );

        // 3. Bogus positional word `know` (from shell quote splitting) must NOT create `/workspace/know`
        let _ =
            run_cli_with_env(&env, &["patch".to_string(), "know".to_string()], "x = 1\n").unwrap();
        assert!(
            !env.ws.join("know").exists(),
            "must never create bogus file `/workspace/know`"
        );
    }

    #[test]
    fn test_dotfile_adk_exec_excluded_and_escaped_quotes_repaired() {
        let (_dir, env) = setup_dummy_git_repo();

        // 1. Simulate ADK subprocess `.adk_exec_999.py` in `/workspace` containing the exact query string
        let adk_script = env.ws.join(".adk_exec_999.py");
        write_lf(&adk_script, "# validate_username allow_dots\n").unwrap();
        let idx = build_repo_index(&env.ws).unwrap();
        assert!(
            !idx.files.iter().any(|f| f.contains(".adk_exec")),
            "dotfiles like .adk_exec_*.py must never be indexed: {:?}",
            idx.files
        );

        // 2. Patch with escaped quotes `\"\"\"` and `[\"alice\"]` outside strings
        let escaped_body = r#"def format_display_name(name: str) -> str:
    \"\"\"Format display name.\"\"\"
    if name in [\"admin\", \"root\"]:
        return name
    return name.strip().lower()
"#;
        let out_esc = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=format_display_name".to_string(),
            ],
            escaped_body,
        )
        .unwrap();
        assert!(out_esc.contains("PATCH APPLIED"), "got: {out_esc}");
        let val_txt = read_lf(&ws_file(&env.ws, "pkg", "validators.py")).unwrap();
        assert!(
            val_txt.contains("\"\"\"Format display name.\"\"\"")
                && val_txt.contains("[\"admin\", \"root\"]"),
            "escaped quotes must be unescaped automatically: {val_txt}"
        );
    }

    #[test]
    fn test_main_block_exclusion_empty_body_and_verify_write_file_unescape() {
        let (_dir, env) = setup_dummy_git_repo();

        // 1. Add `if __name__ == "__main__": class BrokenRepr:` at bottom of `pkg/validators.py`
        let val_path = ws_file(&env.ws, "pkg", "validators.py");
        let mut orig = read_lf(&val_path).unwrap();
        orig.push_str(
            "\nif __name__ == \"__main__\":\n    class BrokenRepr:\n        def __repr__(self):\n            raise Exception(\"demo\")\n",
        );
        write_lf(&val_path, &orig).unwrap();
        let idx = build_repo_index(&env.ws).unwrap();
        assert!(
            !idx.nodes.values().any(|d| d.name == "BrokenRepr"),
            "classes inside `if __name__ == '__main__':` must be excluded from AST index"
        );

        // 2. Empty `body` on `symbol=format_display_name` must be refused without `delete=1`
        let out_empty = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "symbol=format_display_name".to_string(),
            ],
            "   \n",
        )
        .unwrap();
        assert!(
            out_empty.contains("PATCH ERRORS") && out_empty.contains("empty patch"),
            "empty body must be refused: {out_empty}"
        );

        // 3. Simulate `write_file` creating `pkg/commands.py` + `edit_file` introducing `Literal[\"create\"]`
        let new_mod = ws_file(&env.ws, "pkg", "commands.py");
        write_lf(
            &new_mod,
            "from typing import Literal\n\ndef mode() -> Literal[\\\"create\\\"]:\n    return \"create\"\n",
        )
        .unwrap();
        let verify_out = run_cli_with_env(&env, &["verify".to_string()], "").unwrap();
        assert!(
            new_mod.is_file(),
            "`write_file`-created package module `pkg/commands.py` must be preserved by verify"
        );
        assert!(
            verify_out.contains("pkg/commands.py")
                && verify_out.contains("VERDICT: READY TO CALL `submit_patch` NOW."),
            "verify must stage `pkg/commands.py` and auto-unescape `\\\"`: {verify_out}"
        );
        let repaired_mod = read_lf(&new_mod).unwrap();
        assert!(
            repaired_mod.contains("Literal[\"create\"]"),
            "expected auto-unescaped quotes, got: {repaired_mod}"
        );
    }

    #[test]
    fn test_short_ident_localization_overload_critic_and_verify_repro_indent() {
        let (_dir, env) = setup_dummy_git_repo();

        // 1. Asymmetric trailing quotes in `repro=assert f('a') == 'a'` must NOT be stripped
        let (_, parsed_args, _) = parse_cli_and_stdin(
            &[
                "verify".to_string(),
                "repro=import pkg\nassert 'a' == 'a'".to_string(),
            ],
            "",
        );
        assert_eq!(
            parsed_args.get("repro").map(String::as_str),
            Some("import pkg\nassert 'a' == 'a'")
        );

        // 2. Root doc hook `_griffe_ext.py` vs short class module `pkg/h.py` (`H.quantile` pattern)
        write_lf(
            &env.ws.join("_griffe_ext.py"),
            "def on_instance(obj):\n    return obj\n",
        )
        .unwrap();
        write_lf(
            &ws_file(&env.ws, "pkg", "h.py"),
            "from typing import overload\n\nclass H:\n    @overload\n    def items(self, raw: bool) -> list: ...\n    @overload\n    def items(self) -> list: ...\n    def items(self, raw: bool = False) -> list:\n        return []\n",
        )
        .unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&env.ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "add h and hook"])
            .current_dir(&env.ws)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(env.agent_dir.join("index.json"));

        let dbg = run_cli_with_env(
            &env,
            &[
                "debug".to_string(),
                "query=Add H.quantile method to H in pkg/h.py".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            dbg.contains("pkg/h.py") && !dbg.contains("_griffe_ext.py"),
            "debug must rank `pkg/h.py` (`class H`) and exclude root `_griffe_ext.py`, got:\n{dbg}"
        );

        // 3. Adding `quantile` to `class H` (which has `@overload` stubs on `items`) must produce 0 Static Critic warnings
        write_lf(
            &ws_file(&env.ws, "pkg", "h.py"),
            "from typing import overload\n\nclass H:\n    @overload\n    def items(self, raw: bool) -> list: ...\n    @overload\n    def items(self) -> list: ...\n    def items(self, raw: bool = False) -> list:\n        return []\n\n    def quantile(self, q: float) -> float:\n        return q\n",
        )
        .unwrap();
        let critic_clean = patch::run_static_critic(&env, &["pkg/h.py".to_string()]);
        assert!(
            critic_clean.is_empty(),
            "@overload stubs must not trigger false signature warnings: {critic_clean:?}"
        );

        // 4. Deleting existing public function `format_display_name` in `pkg/validators.py` must trigger a BLOCKING critic error
        write_lf(
            &ws_file(&env.ws, "pkg", "validators.py"),
            "import re\n\ndef validate_username(name: str, allow_dots: bool = False) -> bool:\n    return bool(name)\n",
        )
        .unwrap();
        let critic_del = patch::run_static_critic(&env, &["pkg/validators.py".to_string()]);
        assert!(
            critic_del
                .blocking
                .iter()
                .any(|m| m.contains("format_display_name")),
            "deleting public def must be flagged as blocking error: {critic_del:?}"
        );
        // Restore `pkg/validators.py` and test indentation auto-repair + inline `repro=` in `verify`
        Command::new("git")
            .args(["checkout", "HEAD", "--", "pkg/validators.py"])
            .current_dir(&env.ws)
            .output()
            .unwrap();

        // Introduce a +4 space over-indented hunk in `pkg/h.py` (`IndentationError`)
        write_lf(
            &ws_file(&env.ws, "pkg", "h.py"),
            "from typing import overload\n\nclass H:\n    @overload\n    def items(self, raw: bool) -> list: ...\n    @overload\n    def items(self) -> list: ...\n    def items(self, raw: bool = False) -> list:\n        return []\n\n        def quantile(self, q: float) -> float:\n            return q\n",
        )
        .unwrap();
        let v_out = run_cli_with_env(
            &env,
            &[
                "verify".to_string(),
                "repro=from pkg.h import H\nassert H().quantile(0.5) == 0.5".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_out.contains("auto-repaired modified hunk indentation")
                && v_out.contains("REPRODUCER: PASS")
                && v_out.contains("VERDICT: READY TO CALL `submit_patch` NOW."),
            "verify must auto-repair indentation and run inline repro: {v_out}"
        );
    }

    #[test]
    fn test_scoped_old_string_disambiguation_and_safety() {
        let (_tmp, env) = setup_dummy_git_repo();
        assert_eq!(
            view::normalize_symbol_query(
                "def state_command(connection_name: str | None = None) -> None:"
            ),
            "state_command"
        );
        write_lf(
            &ws_file(&env.ws, "pkg", "params.py"),
            "class Param:\n    def __init__(self, x: int = 1):\n        self.val = x\n        self.ok = True\n\nclass Body:\n    def __init__(self, x: int = 1):\n        self.val = x\n        self.ok = True\n",
        )
        .unwrap();

        // 1. Without `symbol`, duplicate `old_string` must report both AST symbols instead of silently editing the first
        let ambig = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/params.py".to_string(),
                "old_string=        self.val = x".to_string(),
                "new_string=        self.val = x + 1".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            ambig.contains("occurs 2 times")
                && ambig.contains("Param.__init__")
                && ambig.contains("Body.__init__"),
            "expected multi-match symbol report, got:\n{ambig}"
        );

        // 2. With `symbol=Body.__init__`, `old_string` disambiguates cleanly on attempt 1
        let ok_scoped = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/params.py".to_string(),
                "symbol=Body.__init__".to_string(),
                "old_string=        self.val = x".to_string(),
                "new_string=        self.val = x + 10".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            ok_scoped.contains("PATCH APPLIED"),
            "scoped old_string must succeed: {ok_scoped}"
        );
        let updated = read_lf(&ws_file(&env.ws, "pkg", "params.py")).unwrap();
        assert!(updated.contains("self.val = x\n") && updated.contains("self.val = x + 10"));

        // 3. Pre-write guard must refuse accidental deletion of `format_display_name`
        let refused_del = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/validators.py".to_string(),
                "old_string=def format_display_name(name: str) -> str:\n    if not validate_username(name):\n        raise ValueError(\"invalid username\")\n    return name.strip().lower()".to_string(),
                "new_string=def new_helper() -> int:\n    return 42".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            refused_del.contains("PATCH ERRORS:") && refused_del.contains("format_display_name"),
            "must refuse accidental public def deletion before writing: {refused_del}"
        );
    }

    #[test]
    fn test_smoke_iter4_indentation_cpython_syntax_and_view_banners() {
        let (_tmp, env) = setup_dummy_git_repo();

        // 1. CPython `ast.parse` catches real `IndentationError` that `tree-sitter-python` accepts
        let bad_indent = "def f(x):\n        y = x + 1\n    return y\n";
        let syn_err = ast_graph::check_python_syntax(bad_indent);
        assert!(
            syn_err.is_some(),
            "check_python_syntax must detect CPython IndentationError"
        );

        // 2. Unindented `old_string` + indented multi-line `new_string` must NOT double-indent line 1
        let router_py = ws_file(&env.ws, "pkg", "router.py");
        write_lf(
            &router_py,
            "from typing import Any\n\nclass APIRouter:\n    def include_router(self, router: Any, prefix: str = \"\") -> None:\n        if prefix:\n            assert prefix.startswith(\"/\")\n",
        )
        .unwrap();
        let stub_py = ws_file(&env.ws, "pkg", "stub.py");
        write_lf(&stub_py, "\"\"\"Deployment model.\"\"\"\n").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&env.ws)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "add router and stub"])
            .current_dir(&env.ws)
            .output()
            .unwrap();

        let patch_out = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/router.py".to_string(),
                "symbol=APIRouter.include_router".to_string(),
                "old_string=if prefix:".to_string(),
                "new_string=        if router is self:\n            raise ValueError(\"self\")\n        if prefix:".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            patch_out.contains("PATCH APPLIED"),
            "expected clean patch without double-indent, got:\n{patch_out}"
        );
        let router_after = read_lf(&router_py).unwrap();
        assert!(
            router_after.contains("\n        if router is self:\n            raise ValueError(\"self\")\n        if prefix:\n"),
            "unexpected indentation in router.py:\n{router_after}"
        );

        // 3. `view` supports `symbol="imports"`, `symbol="*"`, banner-copied symbols, and stub files
        let v_imp = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "file=pkg/router.py".to_string(),
                "symbol=imports".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_imp.contains("module header & imports") && v_imp.contains("from typing import Any")
        );

        let v_banner = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "file=pkg/router.py".to_string(),
                "symbol=pkg/router.py:4-8 (function APIRouter.include_router, file has 8 lines)"
                    .to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_banner.contains("def include_router"),
            "banner symbol must resolve cleanly: {v_banner}"
        );

        // 4. Stub-file upsert (`dokli/models/deployment.py` pattern): `symbol=Service` on 1-line stub file
        let stub_patch = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/stub.py".to_string(),
                "symbol=Service".to_string(),
            ],
            "class Service:\n    name: str = \"app\"\n",
        )
        .unwrap();
        assert!(
            stub_patch.contains("PATCH APPLIED"),
            "stub file upsert must succeed: {stub_patch}"
        );
    }

    #[test]
    fn test_smoke_iter5_after_symbol_toplevel_upsert_and_view_enhancements() {
        let (_tmp_guard, env) = setup_dummy_git_repo();
        let cli_py = env.ws.join("pkg").join("cli.py");
        write_lf(
            &cli_py,
            concat!(
                "def state_command() -> str:\n",
                "    return \"ok\"\n\n",
                "def main() -> None:\n",
                "    print(state_command())\n\n",
                "if __name__ == \"__main__\":\n",
                "    main()\n",
            ),
        )
        .unwrap();

        let types_py = env.ws.join("pkg").join("types.py");
        write_lf(
            &types_py,
            concat!(
                "from typing import Any, Union\n",
                "UnionType = Union[int, str]\n",
                "ModelNameMap = dict[str, str]\n",
            ),
        )
        .unwrap();

        let param_py = env.ws.join("pkg").join("rich_param.py");
        write_lf(
            &param_py,
            concat!(
                "class BaseParam:\n",
                "    def __init__(\n",
                "        self,\n",
                "        name: str,\n",
                "        flag: bool = False,\n",
                "    ) -> None:\n",
                "        self.name = name\n",
                "        self.flag = flag\n\n",
                "class SubOption(BaseParam):\n",
                "    \"\"\"Option subclass.\"\"\"\n",
                "    pass\n",
            ),
        )
        .unwrap();

        // 1. Turn-1 `view` without `debug` emits reminder tip; `file="*"` lists workspace files
        let v_star =
            run_cli_with_env(&env, &["view".to_string(), "file=*".to_string()], "").unwrap();
        assert!(
            v_star.contains("pkg/cli.py")
                && v_star.contains("TIP: You called `view.py` before `scripts/debug.py`"),
            "unexpected file=* output:\n{v_star}"
        );

        // 2. Consecutive duplicate call emits `DUPLICATE CALL` warning
        let v_star_dup =
            run_cli_with_env(&env, &["view".to_string(), "file=*".to_string()], "").unwrap();
        assert!(
            v_star_dup.contains("DUPLICATE CALL (#2"),
            "expected duplicate warning:\n{v_star_dup}"
        );

        // 3. `after_symbol` runs BEFORE `symbol` even when `symbol="init_command"` does not exist yet
        let p_after = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/cli.py".to_string(),
                "symbol=init_command".to_string(),
                "after_symbol=state_command".to_string(),
            ],
            "def init_command() -> str:\n    return \"initialized\"\n",
        )
        .unwrap();
        assert!(
            p_after.contains("PATCH APPLIED") && p_after.contains("inserted after `state_command`"),
            "after_symbol + symbol failed:\n{p_after}"
        );
        // Re-running with updated body updates `init_command` in-place instead of duplicating it
        let p_after_update = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/cli.py".to_string(),
                "symbol=init_command".to_string(),
                "after_symbol=state_command".to_string(),
            ],
            "def init_command() -> str:\n    return \"init-v2\"\n",
        )
        .unwrap();
        assert!(p_after_update.contains("PATCH APPLIED"));
        let cli_after = read_lf(&cli_py).unwrap();
        assert_eq!(
            cli_after.matches("def init_command").count(),
            1,
            "must not duplicate init_command:\n{cli_after}"
        );
        assert!(cli_after.contains("init-v2"));
        // symbol= plus start=/end= with a body that is not a whole def = a line-range edit
        let ln = cli_after
            .lines()
            .position(|l| l.contains("init-v2"))
            .unwrap()
            + 1;
        let p_range = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/cli.py".to_string(),
                "symbol=init_command".to_string(),
                format!("start={ln}"),
                format!("end={ln}"),
            ],
            "    return \"init-v3\"\n",
        )
        .unwrap();
        assert!(
            p_range.contains("PATCH APPLIED"),
            "range edit with symbol failed:\n{p_range}"
        );
        let cli_range = read_lf(&cli_py).unwrap();
        assert!(
            cli_range.contains("init-v3") && cli_range.matches("def init_command").count() == 1,
            "{cli_range}"
        );

        // 4. Additive top-level function upsert via `symbol="deploy_command"` (inserts before `if __name__`)
        let p_top = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/cli.py".to_string(),
                "symbol=deploy_command".to_string(),
            ],
            "def deploy_command() -> str:\n    return \"deployed\"\n",
        )
        .unwrap();
        assert!(
            p_top.contains("PATCH APPLIED")
                && p_top.contains("inserted top-level symbol `deploy_command`"),
            "top-level upsert failed:\n{p_top}"
        );
        let cli_after2 = read_lf(&cli_py).unwrap();
        assert!(
            cli_after2.find("def deploy_command").unwrap()
                < cli_after2.find("if __name__").unwrap(),
            "deploy_command must be placed before if __name__:\n{cli_after2}"
        );

        // 5. Module-level `after_symbol="UnionType"` AND `symbol="pkg/types.py"` preserve existing constants
        let p_types = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/types.py".to_string(),
                "after_symbol=UnionType".to_string(),
            ],
            "def is_alias(tp: Any) -> bool:\n    return tp is not None\n",
        )
        .unwrap();
        assert!(
            p_types.contains("PATCH APPLIED"),
            "after_symbol on assignment failed:\n{p_types}"
        );
        let p_types2 = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/types.py".to_string(),
                "symbol=pkg/types.py".to_string(),
            ],
            "def unwrap_alias(tp: Any) -> Any:\n    return tp\n",
        )
        .unwrap();
        assert!(
            p_types2.contains("PATCH APPLIED"),
            "module-named additive upsert failed:\n{p_types2}"
        );
        let types_after = read_lf(&types_py).unwrap();
        assert!(
            types_after.contains("ModelNameMap")
                && types_after.contains("def is_alias")
                && types_after.contains("def unwrap_alias"),
            "types.py lost constants or new functions:\n{types_after}"
        );

        // 6. Comma-separated `symbol="BaseParam,SubOption"` and inherited `symbol="SubOption.__init__"`
        let v_comma = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "file=pkg/rich_param.py".to_string(),
                "symbol=BaseParam,SubOption".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_comma.contains("class BaseParam") && v_comma.contains("class SubOption"),
            "comma-separated view failed:\n{v_comma}"
        );
        let v_inh = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "file=pkg/rich_param.py".to_string(),
                "symbol=SubOption.__init__".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_inh.contains("does not override `__init__` directly")
                && v_inh.contains("BaseParam.__init__"),
            "inherited method view failed:\n{v_inh}"
        );

        // 7. Dangling multi-line signature `old_string="    def __init__("` emits explicit multi-line signature HINT
        let p_dang = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/rich_param.py".to_string(),
                "old_string=    def __init__(".to_string(),
                "new_string=    def __init__(self, name: str) -> None:".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            p_dang.contains("HINT: `old_string` matched only lines") && p_dang.contains("unclosed"),
            "expected dangling signature hint:\n{p_dang}"
        );
    }

    #[test]
    fn test_smoke_iter6_overload_view_and_patch_guards() {
        let (_tmp, env) = setup_dummy_git_repo();

        let router_py = env.ws.join("pkg").join("router.py");
        write_lf(
            &router_py,
            concat!(
                "from os import PathLike, stat\n",
                "from typing import Annotated, overload\n\n",
                "ServiceDef = Annotated[dict, \"service\"]\n\n",
                "class Router:\n",
                "    @overload\n",
                "    def add_route(self, path: str) -> None: ...\n\n",
                "    @overload\n",
                "    def add_route(self, path: str, *, strict: bool = False) -> None: ...\n\n",
                "    def add_route(\n",
                "        self,\n",
                "        path: str,\n",
                "        strict: bool = False,\n",
                "    ) -> None:\n",
                "        \"\"\"\n",
                "        Line 1\n",
                "        Line 2\n",
                "        Line 3\n",
                "        Line 4\n",
                "        Line 5\n",
                "        Line 6\n",
                "        Line 7\n",
                "        Line 8\n",
                "        Line 9\n",
                "        Line 10\n",
                "        Line 11\n",
                "        Line 12\n",
                "        Line 13\n",
                "        \"\"\"\n",
                "        if path:\n",
                "            assert path.startswith(\"/\")\n",
                "            assert not path.endswith(\"/\")\n",
                "        self._register_internal(path, strict=strict)\n\n",
                "    def _register_internal(self, path: str, strict: bool = False) -> None:\n",
                "        pass\n",
            ),
        )
        .unwrap();
        let _ = Command::new("git")
            .args(["add", "."])
            .current_dir(&env.ws)
            .output();
        let _ = Command::new("git")
            .args(["commit", "-m", "add router"])
            .current_dir(&env.ws)
            .output();

        // 1. `@overload` stubs collapse into the concrete implementation (`view` shows real body, not `...` stub)
        let v_over = run_cli_with_env(
            &env,
            &[
                "view".to_string(),
                "file=pkg/router.py".to_string(),
                "symbol=Router.add_route".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            v_over.contains("_register_internal") && v_over.contains("assert not path.endswith"),
            "view should show concrete @overload implementation, got:\n{v_over}"
        );

        // 2. Bare `symbol=ServiceDef` without `file=` finds module-level TypeAlias assignment
        let v_alias = run_cli_with_env(
            &env,
            &["view".to_string(), "symbol=ServiceDef".to_string()],
            "",
        )
        .unwrap();
        assert!(
            v_alias.contains("ServiceDef = Annotated"),
            "bare symbol TypeAlias lookup failed:\n{v_alias}"
        );

        // 3. Partial compound-block overwrite (`if path:` replaced with `if strict: raise ValueError(...)` leaving
        // `assert not path.endswith("/")` orphaned as dead code!) is refused cleanly
        let p_partial = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/router.py".to_string(),
                "symbol=Router.add_route".to_string(),
                "old_string=        if path:\n            assert path.startswith(\"/\")"
                    .to_string(),
                "new_string=        if strict:\n            raise ValueError(\"bad\")".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            p_partial.contains("refused partial block overwrite"),
            "expected partial block overwrite guard:\n{p_partial}"
        );

        // 4. Statement-prefix mid-line `1C` guard + runtime `ImportError: cannot import name 'BrokenPipe' from 'os'`
        let p_bad_imp = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/router.py".to_string(),
                "old_string=from os import PathLike, stat".to_string(),
                "new_string=from os import PathLike, stat, BrokenPipe".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            p_bad_imp.contains("BrokenPipeError"),
            "expected static critic runtime ImportError + BrokenPipeError hint:\n{p_bad_imp}"
        );

        // Now test that `old_string="from os import PathLike, stat"` does NOT slice `from os import PathLike, stat, BrokenPipe` in half
        let p_no_slice = run_cli_with_env(
            &env,
            &[
                "patch".to_string(),
                "file=pkg/router.py".to_string(),
                "old_string=from os import PathLike, stat".to_string(),
                "new_string=from os import PathLike, stat\nimport errno".to_string(),
            ],
            "",
        )
        .unwrap();
        assert!(
            p_no_slice.contains("did not match"),
            "statement prefix should NOT slice `from os import PathLike, stat, BrokenPipe` in half:\n{p_no_slice}"
        );
    }

    #[test]
    fn test_pillar3_multi_file_graph_and_determinism() {
        let (_tmp, env) = setup_dummy_git_repo();

        let base_py = env.ws.join("pkg").join("base.py");
        write_lf(
            &base_py,
            concat!(
                "class BaseOption:\n",
                "    def process(self, value: str) -> str:\n",
                "        return value.strip()\n",
            ),
        )
        .unwrap();

        let rich_py = env.ws.join("pkg").join("rich_option.py");
        write_lf(
            &rich_py,
            concat!(
                "from .base import BaseOption\n\n",
                "class RichOption(BaseOption):\n",
                "    def process(self, value: str) -> str:\n",
                "        return super().process(value).lower()\n",
            ),
        )
        .unwrap();

        let init_py = env.ws.join("pkg").join("__init__.py");
        write_lf(
            &init_py,
            concat!(
                "from .validators import validate_username, AsyncValidator\n",
                "from .rich_option import RichOption\n",
            ),
        )
        .unwrap();

        let _ = Command::new("git")
            .args(["add", "."])
            .current_dir(&env.ws)
            .output();
        let _ = Command::new("git")
            .args(["commit", "-m", "add pillar 3 fixtures"])
            .current_dir(&env.ws)
            .output();

        let idx = load_or_build_index(&env).expect("index");
        // TOP TARGETS give every file one slot before a file gets its second
        let mut by_file: std::collections::BTreeMap<&str, Vec<&String>> = Default::default();
        for (id, n) in &idx.nodes {
            by_file.entry(n.file.as_str()).or_default().push(id);
        }
        let mut files = by_file.values().filter(|v| v.len() >= 2);
        let a = files.next().expect("a file with 2 defs");
        let b = by_file
            .values()
            .find(|v| v[0] != a[0])
            .expect("a second file");
        let ranked = vec![
            (3.0, a[0].clone()),
            (2.0, a[1].clone()),
            (1.0, b[0].clone()),
        ];
        let div = crate::debug::diversify_by_file(&idx, ranked, 3);
        let order: Vec<&str> = div.iter().map(|(_, id)| id.as_str()).collect();
        assert_eq!(order, vec![a[0].as_str(), b[0].as_str(), a[1].as_str()]);
        // 1. Base <-> Subclass bi-directional graph
        let base_id = "pkg.base.BaseOption";
        let rich_id = "pkg.rich_option.RichOption";
        assert!(
            idx.subclasses
                .get(base_id)
                .map(|s| s.contains(&rich_id.to_string()))
                .unwrap_or(false),
            "subclasses missing rich_id: {:?}",
            idx.subclasses.get(base_id)
        );
        assert!(
            idx.superclasses
                .get(rich_id)
                .map(|s| s.contains(&base_id.to_string()))
                .unwrap_or(false),
            "superclasses missing base_id: {:?}",
            idx.superclasses.get(rich_id)
        );

        // 2. Method override bi-directional edges
        let base_m = "pkg.base.BaseOption.process";
        let rich_m = "pkg.rich_option.RichOption.process";
        assert!(
            idx.adj
                .get(rich_m)
                .map(|s| s.contains(&base_m.to_string()))
                .unwrap_or(false),
            "adj missing method override edge: {:?}",
            idx.adj.get(rich_m)
        );

        // 3. ALSO-CHANGE multi-file discovery finds both base.py and __init__.py
        let d1 = run_cli_with_env(
            &env,
            &["debug".to_string(), "query=RichOption.process".to_string()],
            "",
        )
        .unwrap();
        assert!(
            d1.contains("ALSO-CHANGE")
                && (d1.contains("pkg/base.py") || d1.contains("pkg/__init__.py")),
            "debug ALSO-CHANGE failed to discover related files:\n{d1}"
        );

        // 4. Determinism invariant: two sequential debug calls produce byte-for-byte identical core output
        let _ = std::fs::remove_file(env.agent_dir.join("last_call.json"));
        let d2 = run_cli_with_env(
            &env,
            &["debug".to_string(), "query=RichOption.process".to_string()],
            "",
        )
        .unwrap();
        assert_eq!(d1, d2, "debug output is non-deterministic!");
    }

    #[test]
    fn role_words_add_operation_names_to_the_query() {
        // "the writing part" must reach `import_toml_dumps`
        let (_, words, _) = crate::debug::extract_query_terms(
            "only the reading part has been integrated, not the writing part of toml",
        );
        for w in ["dumps", "dump", "loads", "load"] {
            assert!(words.iter().any(|x| x == w), "{w} missing from {words:?}");
        }
        let (_, plain, _) = crate::debug::extract_query_terms("Fix traceback recursion in Console");
        assert!(!plain.iter().any(|x| x == "dumps" || x == "loads"));
    }

    #[test]
    fn budget_scale_follows_max_time() {
        // SAFETY: test-local variables; no other test reads SWE_MAX_TIME_S / SWE_MAX_TOOL_CALLS
        unsafe {
            std::env::set_var("SWE_MAX_TIME_S", "480");
            std::env::set_var("SWE_MAX_TOOL_CALLS", "28");
        }
        assert!((crate::budget_scale() - 480.0 / 288.0).abs() < 1e-9);
        assert_eq!(crate::max_tool_calls(), 28);
        unsafe {
            std::env::remove_var("SWE_MAX_TIME_S");
            std::env::remove_var("SWE_MAX_TOOL_CALLS");
        }
        assert_eq!(crate::max_tool_calls(), 26);
    }
}
