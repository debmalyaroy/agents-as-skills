//! `/tmp/coder debug`: AST/graph/embedding localization, five-signal `ALSO-CHANGE`
//! counterpart discovery, reproducer run and preloaded numbered source slices.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::ast_graph::{
    DefNode, RepoIndex, find_structural_sibling, is_docs_path, is_test_path,
    tokenize_ident_and_subwords,
};
use crate::{
    EnvPaths, clip_tokens, format_numbered_slice, load_or_build_index, run_shell_with_timeout,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReproOutcome {
    pub has_repro: bool,
    pub code: String,
    pub fails_now: bool,
    pub summary: String,
    /// `("pkg/mod.py", line_no, "func_name")`
    pub frames: Vec<(String, usize, String)>,
}

/// Extract runnable Python snippets (fenced ```python blocks or `>>>` doctests) from the issue text.
pub fn extract_repro_candidates(issue: &str) -> Vec<String> {
    let fence_re = Regex::new(r"(?s)```[ \t]*(\w*)[^\n]*\n(.*?)```").expect("regex");
    let not_py_re =
        Regex::new(r"(?m)^\s*(\$ |pip |uv |poetry |python -m pip|curl |git |docker |npm |pytest )")
            .expect("regex");
    let shell_langs: HashSet<&str> = [
        "bash",
        "sh",
        "shell",
        "console",
        "text",
        "txt",
        "toml",
        "yaml",
        "yml",
        "json",
        "ini",
        "cfg",
        "html",
        "js",
        "diff",
        "pycon_output",
    ]
    .into_iter()
    .collect();

    let mut found = Vec::new();
    for cap in fence_re.captures_iter(issue) {
        let lang = cap
            .get(1)
            .map(|m| m.as_str().to_ascii_lowercase())
            .unwrap_or_default();
        if shell_langs.contains(lang.as_str()) {
            continue;
        }
        let body = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let code = if body.lines().any(|l| l.trim_start().starts_with(">>> ")) {
            extract_doctest_lines(body)
        } else {
            body.trim().to_string()
        };
        if code.is_empty()
            || code.trim_start().starts_with("Traceback")
            || code.trim_start().starts_with('$')
            || not_py_re.is_match(&code)
        {
            continue;
        }
        if crate::ast_graph::check_python_syntax(&code).is_none() && !found.contains(&code) {
            found.push(code);
        }
    }
    if found.is_empty() {
        let doc = extract_doctest_lines(issue);
        if !doc.is_empty() && crate::ast_graph::check_python_syntax(&doc).is_none() {
            found.push(doc);
        }
    }
    found.into_iter().take(3).collect()
}

fn extract_doctest_lines(block: &str) -> String {
    let mut out = Vec::new();
    for ln in block.lines() {
        let s = ln.trim_start();
        if let Some(rest) = s.strip_prefix(">>> ").or_else(|| s.strip_prefix("... ")) {
            out.push(rest);
        } else if s == ">>>" || s == "..." {
            out.push("");
        }
    }
    out.join("\n").trim().to_string()
}

/// Parse traceback frames inside `ws` from Python stderr/stdout using OS-agnostic `Path` matching.
pub fn parse_repo_frames(output: &str, ws: &Path) -> Vec<(String, usize, String)> {
    let frame_re = Regex::new(r#"File "([^"]+)", line (\d+), in (\S+)"#).expect("regex");
    let mut out = Vec::new();
    for cap in frame_re.captures_iter(output) {
        let raw_p = Path::new(&cap[1]);
        let line_no: usize = cap[2].parse().unwrap_or(0);
        let fn_name = cap[3].to_string();
        let rel = if let Ok(stripped) = raw_p.strip_prefix(ws) {
            crate::clean_rel_arg(&stripped.to_string_lossy())
        } else {
            let cand_rel = crate::clean_rel_arg(&cap[1]);
            if !cand_rel.is_empty() && crate::join_rel(ws, &cand_rel).is_file() {
                cand_rel
            } else {
                continue;
            }
        };
        let segs = crate::path_segments(&rel);
        if segs.iter().any(|s| s == "probe") || is_test_path(&rel) {
            continue;
        }
        let item = (rel, line_no, fn_name);
        if !out.contains(&item) {
            out.push(item);
        }
    }
    let len = out.len();
    if len > 4 { out.split_off(len - 4) } else { out }
}

/// Try running extracted reproducer candidates in `/tmp/.agent/probe/` (`PYTHONPATH=/workspace`).
pub fn run_or_load_repro(env: &EnvPaths, issue: &str, index: &RepoIndex) -> ReproOutcome {
    let repro_meta_path = env.agent_dir.join("repro_meta.json");
    if repro_meta_path.is_file()
        && let Ok(txt) = crate::read_lf(&repro_meta_path)
        && let Ok(cached) = serde_json::from_str::<ReproOutcome>(&txt)
    {
        return cached;
    }

    // Also extract traceback frames directly mentioned in the issue text itself
    let mut outcome = ReproOutcome {
        frames: parse_repo_frames(issue, &env.ws),
        ..Default::default()
    };

    let cands = extract_repro_candidates(issue);
    if cands.is_empty() {
        let _ = crate::write_lf(
            &repro_meta_path,
            &serde_json::to_string(&outcome).unwrap_or_default(),
        );
        return outcome;
    }

    let top_mods: HashSet<String> = index
        .files
        .iter()
        .filter_map(|f| {
            crate::path_segments(f)
                .into_iter()
                .next()
                .map(|s| s.trim_end_matches(".py").to_string())
        })
        .collect();

    let probe_dir = env.agent_dir.join("probe");
    let _ = std::fs::create_dir_all(&probe_dir);
    let exc_re = Regex::new(
        r"(?m)^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Exception|Warning|Exit|Interrupt))\b[:]?\s*(.*)$",
    )
    .expect("regex");
    let missing_mod_re =
        Regex::new(r"ModuleNotFoundError: No module named '([\w.]+)'").expect("regex");

    for (i, code) in cands.iter().enumerate() {
        let script_path = probe_dir.join(format!("auto_{i}.py"));
        if crate::write_lf(&script_path, &format!("{code}\n")).is_err() {
            continue;
        }
        let sp = script_path.display().to_string();
        let cmd = if sp.contains(' ') {
            format!("python3 \"{sp}\"")
        } else {
            format!("python3 {sp}")
        };
        let (rc, out) = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, 12);
        if rc == 0 || rc == 124 {
            continue;
        }
        if let Some(cap) = missing_mod_re.captures(&out) {
            let root_mod = cap[1].split('.').next().unwrap_or("");
            if !top_mods.contains(root_mod) {
                continue; // 3rd-party optional package missing in sandbox
            }
        }
        let mut last_exc = None;
        for cap in exc_re.captures_iter(&out) {
            let name = cap[1].to_string();
            let msg = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if name == "SyntaxError" || name == "IndentationError" {
                continue;
            }
            last_exc = Some(format!(
                "{name}: {}",
                msg.chars().take(140).collect::<String>()
            ));
        }
        if let Some(summary) = last_exc {
            let frames = parse_repo_frames(&out, &env.ws);
            let saved_repro = env.agent_dir.join("repro.py");
            let _ = crate::write_lf(&saved_repro, &format!("{code}\n"));
            outcome = ReproOutcome {
                has_repro: true,
                code: code.chars().take(800).collect(),
                fails_now: true,
                summary,
                frames: if frames.is_empty() {
                    outcome.frames
                } else {
                    frames
                },
            };
            break;
        }
    }

    let _ = crate::write_lf(
        &repro_meta_path,
        &serde_json::to_string(&outcome).unwrap_or_default(),
    );
    outcome
}

const QUERY_STOP: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "should",
    "not",
    "are",
    "add",
    "support",
    "use",
    "new",
    "fix",
    "bug",
    "issue",
    "error",
    "return",
    "value",
    "none",
    "true",
    "false",
    "def",
    "class",
    "self",
    "import",
    "file",
    "code",
    "test",
    "tests",
    "http",
    "https",
    "www",
    "com",
    "org",
    "github",
    "pull",
    "discussions",
    "referenced",
    "solution",
    "python",
    "related",
    "see",
    "thanks",
    "fixes",
    "closes",
    "doing",
    "type",
    "changes",
    "feature",
    "documentation",
    "docstrings",
    "other",
    "checklist",
    "latest",
    "black",
    "default",
    "updated",
    "changelog",
    "contributors",
    "where",
    "appropriate",
    "added",
];

fn clean_issue_boilerplate(text: &str) -> String {
    let url_re = Regex::new(r"https?://\S+|www\.\S+").expect("regex");
    let no_urls = url_re.replace_all(text, " ");
    let mut kept = Vec::new();
    for line in no_urls.lines() {
        let t = line.trim();
        if t.starts_with("- [ ]") || t.starts_with("- [x]") || t.starts_with("- [X]") {
            continue;
        }
        kept.push(line);
    }
    kept.join("\n")
}

/// Extract important identifiers, dotted paths, and file paths from the query/issue text.
pub fn extract_query_terms(text: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let cleaned = clean_issue_boilerplate(text);
    let file_re = Regex::new(r"\b([A-Za-z0-9_./-]+\.py)\b").expect("regex");
    let phrase_re = Regex::new(r#"['"`]([^'"`\n]{3,80})['"`]"#).expect("regex");
    let word_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_.]{2,}").expect("regex");

    let mut explicit_files = Vec::new();
    for cap in file_re.captures_iter(&cleaned) {
        let f = cap[1].trim_start_matches("./").to_string();
        if !explicit_files.contains(&f) {
            explicit_files.push(f);
        }
    }

    let mut phrases = Vec::new();
    for cap in phrase_re.captures_iter(&cleaned) {
        let p = cap[1].trim().to_string();
        if p.len() >= 3 && !phrases.contains(&p) {
            phrases.push(p);
            if phrases.len() >= 5 {
                break;
            }
        }
    }

    let mut words = Vec::new();
    for m in word_re.find_iter(&cleaned) {
        let w = m.as_str();
        for part in std::iter::once(w).chain(w.split('.')) {
            let lw = part.to_ascii_lowercase();
            if lw.len() >= 3
                && !lw.chars().all(|c| c.is_ascii_digit())
                && !QUERY_STOP.contains(&lw.as_str())
                && !words.contains(&lw)
            {
                words.push(lw);
                if words.len() >= 24 {
                    break;
                }
            }
        }
        if words.len() >= 24 {
            break;
        }
    }
    // Also add CamelCase subwords & adjacent CamelCase compound bigrams (e.g. TypeAliasType -> type_alias, BrokenPipe -> broken_pipe)
    let ident_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").expect("regex");
    for m in ident_re.find_iter(&cleaned) {
        let parts = split_camel_parts(m.as_str());
        if parts.len() >= 2 {
            for win in parts.windows(2) {
                for cand in [
                    format!("{}_{}", win[0], win[1]),
                    format!("{}{}", win[0], win[1]),
                ] {
                    if cand.len() >= 5 && !words.contains(&cand) && words.len() < 30 {
                        words.push(cand);
                    }
                }
            }
        }
    }
    for sub in tokenize_ident_and_subwords(&cleaned) {
        if sub.len() >= 3 && !QUERY_STOP.contains(&sub.as_str()) && !words.contains(&sub) {
            words.push(sub);
            if words.len() >= 32 {
                break;
            }
        }
    }
    // Issue prose names the role ("the _writing_ part"), code names the operation (`dumps`):
    // expand role words to the operation names before ranking.
    let prose: BTreeSet<String> = word_re
        .find_iter(&cleaned.to_ascii_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    for (triggers, ops) in ROLE_SYNONYMS {
        if triggers.iter().any(|t| prose.contains(*t)) {
            for op in ops {
                if !words.iter().any(|w| w == op) {
                    words.push((*op).to_string());
                }
            }
        }
    }
    (explicit_files, words, phrases)
}

/// Role words in issue prose -> the operation names code uses for them.
pub const ROLE_SYNONYMS: [(&[&str], &[&str]); 2] = [
    (
        &[
            "write",
            "writing",
            "writer",
            "written",
            "serialize",
            "serializing",
            "serialization",
            "saving",
            "export",
            "exporting",
        ],
        &["dump", "dumps", "write", "save"],
    ),
    (
        &[
            "read",
            "reading",
            "reader",
            "parse",
            "parsing",
            "deserialize",
            "deserializing",
            "loading",
        ],
        &["load", "loads", "read", "parse"],
    ),
];

fn split_camel_parts(ident: &str) -> Vec<String> {
    let mut out = Vec::new();
    for seg in ident.split('_').filter(|s| !s.is_empty()) {
        let chars: Vec<char> = seg.chars().collect();
        let mut start = 0usize;
        for i in 1..chars.len() {
            let prev = chars[i - 1];
            let cur = chars[i];
            let next_is_lower = chars.get(i + 1).is_some_and(|c| c.is_ascii_lowercase());
            if (prev.is_ascii_lowercase() && cur.is_ascii_uppercase())
                || (prev.is_ascii_uppercase() && cur.is_ascii_uppercase() && next_is_lower)
            {
                let part: String = chars[start..i].iter().collect();
                out.push(part.to_ascii_lowercase());
                start = i;
            }
        }
        let tail: String = chars[start..].iter().collect();
        if !tail.is_empty() {
            out.push(tail.to_ascii_lowercase());
        }
    }
    out
}

/// Identify dominant repository package root tokens (e.g. `fastapi`, `rich_click`, `rich`, `click`, `httpx`)
/// that appear in `>= 25%` of source file paths so generic repo names do not drown out high-IDF domain terms.
fn dominant_package_tokens(index: &RepoIndex) -> HashSet<String> {
    let mut pkg_counts: HashMap<String, usize> = HashMap::new();
    let mut n_src = 0usize;
    for f in &index.files {
        if is_test_path(f) || is_docs_path(f) {
            continue;
        }
        let segs = crate::path_segments(&f.to_ascii_lowercase());
        if segs.len() < 2 {
            continue;
        }
        n_src += 1;
        let pkg = if segs[0] == "src" && segs.len() >= 3 {
            &segs[1]
        } else {
            &segs[0]
        };
        *pkg_counts.entry(pkg.clone()).or_insert(0) += 1;
    }
    let mut out = HashSet::new();
    if n_src >= 4 {
        for (pkg, cnt) in pkg_counts {
            if cnt * 4 >= n_src {
                for part in pkg.split('_').filter(|p| p.len() >= 3) {
                    out.insert(part.to_string());
                }
                out.insert(pkg);
            }
        }
    }
    out
}

/// Extract dotted `LHS.RHS` symbol pairs (including 1–2 char class/module prefixes like `H.quantile`)
/// and standalone 1–2 char uppercase class identifiers (e.g. `class H`, `` `H` ``).
fn extract_dotted_and_short_idents(text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let cleaned = clean_issue_boilerplate(text);
    let dot_re =
        Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b").expect("regex");
    let upper_re = Regex::new(r"(?:`|\bclass\s+|\b)([A-Z][A-Za-z0-9_]{0,1})\b").expect("regex");

    let mut pairs = Vec::new();
    let mut short_idents = Vec::new();
    for cap in dot_re.captures_iter(&cleaned) {
        let lhs = cap[1].to_ascii_lowercase();
        let rhs = cap[2].to_ascii_lowercase();
        if rhs == "py" || QUERY_STOP.contains(&lhs.as_str()) {
            continue;
        }
        let pair = (lhs.clone(), rhs);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
        if lhs.len() <= 2 && !short_idents.contains(&lhs) {
            short_idents.push(lhs);
        }
    }
    for cap in upper_re.captures_iter(&cleaned) {
        let s = cap[1].to_ascii_lowercase();
        if !s.is_empty()
            && s.len() <= 2
            && !QUERY_STOP.contains(&s.as_str())
            && !short_idents.contains(&s)
        {
            short_idents.push(s);
        }
    }
    (pairs, short_idents)
}

/// Linear file ranker over lexical features, plus multi-term coverage, dotted-pair and traceback boosts.
pub fn rank_files_dm1(
    ws: &Path,
    index: &RepoIndex,
    query: &str,
    repro: &ReproOutcome,
) -> Vec<(f64, String)> {
    let (explicit_files, words, phrases) = extract_query_terms(query);
    let (dotted_pairs, short_idents) = extract_dotted_and_short_idents(query);
    let pkg_tokens = dominant_package_tokens(index);

    let mut file_texts: HashMap<&str, (String, String, usize)> = HashMap::new();
    let mut src_df: HashMap<&str, usize> = HashMap::new();
    for f in &index.files {
        if let Ok(t) = crate::read_lf(&crate::join_rel(ws, f)) {
            let n_lines = t.bytes().filter(|&b| b == b'\n').count() + 1;
            let low = t.to_ascii_lowercase();
            file_texts.insert(f.as_str(), (t, low, n_lines));
        }
    }
    let n_files = file_texts.len().max(1) as f64;
    let mut df: HashMap<&str, usize> = HashMap::new();
    for w in &words {
        let mut count_all = 0usize;
        let mut count_src = 0usize;
        for (fp, (_, low, _)) in &file_texts {
            if low.contains(w.as_str()) {
                count_all += 1;
                if !is_test_path(fp) && !is_docs_path(fp) {
                    count_src += 1;
                }
            }
        }
        df.insert(w.as_str(), count_all);
        src_df.insert(w.as_str(), count_src);
    }

    let frame_files: HashSet<&str> = repro.frames.iter().map(|(f, _, _)| f.as_str()).collect();
    let files_low: HashSet<String> = index.files.iter().map(|f| f.to_ascii_lowercase()).collect();

    let mut scored = Vec::with_capacity(index.files.len());
    for f in &index.files {
        let Some((txt, low_txt, n_lines)) = file_texts.get(f.as_str()) else {
            continue;
        };
        let low_path = f.to_ascii_lowercase();
        let segs = crate::path_segments(&low_path);
        let base = segs.last().map(String::as_str).unwrap_or("");
        let base_stem = base.trim_end_matches(".py");
        let dir_segs = if segs.len() > 1 {
            &segs[..segs.len() - 1]
        } else {
            &[]
        };

        let mut idf_sum = 0.0f64;
        let mut distinct_hits = 0usize;
        let mut rare_needle_bonus = 0.0f64;
        for w in &words {
            if w.contains('.') {
                continue;
            }
            let c = low_txt.matches(w.as_str()).count();
            if c > 0 {
                let is_pkg = pkg_tokens.contains(w.as_str());
                if !is_pkg {
                    distinct_hits += 1;
                }
                let dfw = (*df.get(w.as_str()).unwrap_or(&0)) as f64;
                let w_mult = if is_pkg { 0.15 } else { 1.0 };
                idf_sum += w_mult * (1.0 + n_files / (1.0 + dfw)).ln() * (1.0 + (c as f64).ln());
                if !is_pkg && w.len() >= 4 {
                    let sdf = *src_df.get(w.as_str()).unwrap_or(&0);
                    if (1..=3).contains(&sdf) {
                        rare_needle_bonus += 12.0;
                    } else if (4..=7).contains(&sdf) {
                        rare_needle_bonus += 5.0;
                    }
                }
            }
        }
        rare_needle_bonus = rare_needle_bonus.min(28.0);

        let phrase_hits = phrases
            .iter()
            .filter(|p| txt.contains(p.as_str()))
            .count()
            .min(4) as f64;

        let name_hit: f64 = words
            .iter()
            .filter(|w| !w.contains('.') && base_stem.contains(w.as_str()))
            .map(|w| {
                if pkg_tokens.contains(w.as_str()) {
                    0.15
                } else {
                    1.0
                }
            })
            .sum();
        let exact_stem_hit = words
            .iter()
            .chain(short_idents.iter())
            .any(|w| base_stem == w.as_str() && !pkg_tokens.contains(w.as_str()));
        let dir_hit: f64 = words
            .iter()
            .filter(|w| !w.contains('.') && dir_segs.iter().any(|d| d.contains(w.as_str())))
            .map(|w| {
                if pkg_tokens.contains(w.as_str()) {
                    0.1
                } else {
                    1.0
                }
            })
            .sum();

        let defs_in_file = index.file_defs.get(f).map(|v| v.as_slice()).unwrap_or(&[]);
        let def_nodes: Vec<&DefNode> = defs_in_file
            .iter()
            .filter_map(|id| index.nodes.get(id))
            .collect();

        let def_hits: f64 = def_nodes
            .iter()
            .filter_map(|d| {
                let matched = words
                    .iter()
                    .chain(short_idents.iter())
                    .find(|s| d.name.eq_ignore_ascii_case(s) || d.qualname.eq_ignore_ascii_case(s));
                matched.map(|s| {
                    if pkg_tokens.contains(s.as_str()) {
                        0.15
                    } else {
                        1.0
                    }
                })
            })
            .sum::<f64>()
            .min(5.0);

        let mut dotted_bonus = 0.0f64;
        for (lhs, rhs) in &dotted_pairs {
            let defines_lhs = def_nodes
                .iter()
                .any(|d| d.name.eq_ignore_ascii_case(lhs) || d.qualname.eq_ignore_ascii_case(lhs));
            if defines_lhs || base_stem == lhs {
                dotted_bonus += if low_txt.contains(rhs.as_str()) {
                    24.0
                } else {
                    16.0
                };
            }
        }

        if distinct_hits == 0
            && phrase_hits == 0.0
            && name_hit == 0.0
            && !exact_stem_hit
            && dir_hit == 0.0
            && def_hits == 0.0
            && dotted_bonus == 0.0
            && !frame_files.contains(f.as_str())
            && !explicit_files
                .iter()
                .any(|ef| Path::new(f).ends_with(Path::new(ef)))
        {
            continue;
        }

        let is_test = if is_test_path(f) { 1.0 } else { 0.0 };
        let is_docs = if is_docs_path(f) { 1.0 } else { 0.0 };
        let log_size = (1.0 + (*n_lines as f64)).ln();
        let depth = dir_segs.len() as f64;
        let init_file = if base == "__init__.py" { 1.0 } else { 0.0 };
        let coverage_bonus = 5.0 * (distinct_hits.saturating_sub(1) as f64);

        // Linear combination of the lexical features, plus multi-term, needle and dotted-pair signals
        let mut score =
            0.7152 * idf_sum + 14.3697 * name_hit + 8.9816 * phrase_hits + 4.5 * def_hits
                - 38.004 * is_test
                - 25.0 * is_docs
                + 0.3581 * log_size
                - 1.6026 * depth
                - 2.0 * init_file
                + 7.2278 * dir_hit
                + coverage_bonus
                + rare_needle_bonus
                + dotted_bonus
                + if exact_stem_hit { 10.0 } else { 0.0 };

        // A legacy copy (`_compat/v1.py`) ranks below its current form (`v2.py`), which is the code
        // the installed versions import.
        if is_legacy_twin(&low_path, &files_low) && !legacy_named_in(query) {
            score -= 12.0;
        }
        if frame_files.contains(f.as_str()) {
            score += 40.0;
        }
        if explicit_files
            .iter()
            .any(|ef| Path::new(f).ends_with(Path::new(ef)) || low_path == ef.to_ascii_lowercase())
        {
            score += 30.0;
        }

        scored.push((score, f.clone()));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// A legacy copy of code that also exists in a current form: a `v1` path whose `v2` sibling is in the
/// index, or a file under `legacy/`/`deprecated/` whose file name also exists outside it.
pub fn is_legacy_twin(low_path: &str, files_low: &HashSet<String>) -> bool {
    let segs: Vec<&str> = low_path.split('/').collect();
    for (i, s) in segs.iter().enumerate() {
        let stem = s.trim_end_matches(".py");
        if stem == "v1" || stem.ends_with("_v1") {
            let mut twin = segs.clone();
            let repl = s.replacen("v1", "v2", 1);
            twin[i] = repl.as_str();
            if files_low.contains(&twin.join("/")) {
                return true;
            }
        }
        if (*s == "legacy" || *s == "deprecated") && i + 1 < segs.len() {
            let name = segs[segs.len() - 1];
            if files_low.iter().any(|g| {
                g != low_path
                    && g.rsplit('/').next() == Some(name)
                    && !g.contains("/legacy/")
                    && !g.contains("/deprecated/")
            }) {
                return true;
            }
        }
    }
    false
}

fn legacy_named_in(query: &str) -> bool {
    let q = query.to_ascii_lowercase();
    ["v1", "legacy", "deprecated", "compat"]
        .iter()
        .any(|w| q.contains(w))
}

/// Keep the ranked order but give each file one slot before any file gets a second, so several
/// definitions of one file cannot fill TOP TARGETS.
pub fn diversify_by_file(
    index: &RepoIndex,
    ranked: Vec<(f64, String)>,
    top_k: usize,
) -> Vec<(f64, String)> {
    let file_of = |id: &str| {
        index
            .nodes
            .get(id)
            .map(|n| n.file.clone())
            .unwrap_or_default()
    };
    let mut seen: HashSet<String> = HashSet::new();
    let (mut first, mut rest): (Vec<_>, Vec<_>) = (Vec::new(), Vec::new());
    for item in ranked {
        if seen.insert(file_of(&item.1)) {
            first.push(item);
        } else {
            rest.push(item);
        }
    }
    first.into_iter().chain(rest).take(top_k).collect()
}

/// Function-level BM25 + 1-hop call/inheritance graph propagation + reproducer frame boost.
pub fn rank_functions_graph(
    index: &RepoIndex,
    query: &str,
    file_rank: &[(f64, String)],
    repro: &ReproOutcome,
    top_k: usize,
) -> Vec<(f64, String)> {
    let (_, symbols, _) = extract_query_terms(query);
    let (dotted_pairs, short_idents) = extract_dotted_and_short_idents(query);
    let pkg_tokens = dominant_package_tokens(index);
    let mut sym_low: HashSet<String> = symbols.iter().map(|s| s.to_ascii_lowercase()).collect();
    for s in &short_idents {
        sym_low.insert(s.clone());
    }
    for (lhs, rhs) in &dotted_pairs {
        sym_low.insert(lhs.clone());
        sym_low.insert(rhs.clone());
        sym_low.insert(format!("{lhs}.{rhs}"));
    }

    let top_files: HashMap<&str, usize> = file_rank
        .iter()
        .filter(|(_, f)| !is_test_path(f) && !is_docs_path(f))
        .take(18)
        .enumerate()
        .map(|(i, (_, f))| (f.as_str(), i))
        .collect();

    let mut pool: BTreeSet<String> = BTreeSet::new();
    for (id, node) in &index.nodes {
        if is_test_path(&node.file) || is_docs_path(&node.file) {
            continue;
        }
        if top_files.contains_key(node.file.as_str())
            || sym_low.contains(&node.name.to_ascii_lowercase())
            || sym_low.contains(&node.qualname.to_ascii_lowercase())
        {
            pool.insert(id.clone());
            if let Some(nbs) = index.adj.get(id) {
                for nb in nbs {
                    if index
                        .nodes
                        .get(nb)
                        .is_some_and(|n| !is_test_path(&n.file) && !is_docs_path(&n.file))
                    {
                        pool.insert(nb.clone());
                    }
                }
            }
        }
    }
    if pool.is_empty() {
        let mut fallback_nodes: Vec<&DefNode> = index
            .nodes
            .values()
            .filter(|n| !is_test_path(&n.file) && !is_docs_path(&n.file))
            .collect();
        fallback_nodes.sort_by_key(|n| std::cmp::Reverse(n.end_line.saturating_sub(n.start_line)));
        for node in fallback_nodes.into_iter().take(50) {
            pool.insert(node.id.clone());
        }
    }
    if pool.is_empty() {
        return Vec::new();
    }

    let cleaned_q = clean_issue_boilerplate(query);
    let q_toks = tokenize_ident_and_subwords(&cleaned_q);
    let mut q_tf: HashMap<String, usize> = HashMap::new();
    for t in q_toks {
        if !QUERY_STOP.contains(&t.as_str()) {
            *q_tf.entry(t).or_insert(0) += 1;
        }
    }
    for s in &short_idents {
        *q_tf.entry(s.clone()).or_insert(0) += 1;
    }

    let n_docs = pool.len() as f64;
    let avg_len = pool
        .iter()
        .filter_map(|id| index.nodes.get(id))
        .map(|n| n.tokens.len() as f64)
        .sum::<f64>()
        / n_docs.max(1.0);

    let mut df: HashMap<&str, usize> = HashMap::new();
    for id in &pool {
        if let Some(node) = index.nodes.get(id) {
            let unique: HashSet<&str> = node.tokens.iter().map(|s| s.as_str()).collect();
            for t in unique {
                if q_tf.contains_key(t) {
                    *df.entry(t).or_insert(0) += 1;
                }
            }
        }
    }

    let mut bm25_base: HashMap<String, f64> = HashMap::new();
    for id in &pool {
        let Some(node) = index.nodes.get(id) else {
            continue;
        };
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in &node.tokens {
            if q_tf.contains_key(t.as_str()) {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
        }
        let d_len = node.tokens.len() as f64;
        let mut s = 0.0f64;
        for (t, &c) in &tf {
            let dft = (*df.get(t).unwrap_or(&1)) as f64;
            let idf = (1.0 + (n_docs - dft + 0.5) / (dft + 0.5)).ln();
            let cf = c as f64;
            s += idf * (cf * 2.2) / (cf + 1.2 * (0.25 + 0.75 * d_len / avg_len.max(1.0)));
        }
        bm25_base.insert(id.clone(), s);
    }

    let mut results = Vec::with_capacity(pool.len());
    for id in &pool {
        let Some(node) = index.nodes.get(id) else {
            continue;
        };
        let base = *bm25_base.get(id).unwrap_or(&0.0);
        let nbs: Vec<f64> = index
            .adj
            .get(id)
            .into_iter()
            .flatten()
            .filter_map(|nb| bm25_base.get(nb).copied())
            .collect();
        let prop = if nbs.is_empty() {
            0.0
        } else {
            nbs.iter().sum::<f64>() / ((1.0 + nbs.len() as f64).sqrt())
        };
        let node_name_low = node.name.to_ascii_lowercase();
        let node_qual_low = node.qualname.to_ascii_lowercase();
        let name_hit = if sym_low.contains(&node_name_low) || sym_low.contains(&node_qual_low) {
            if pkg_tokens.contains(&node_name_low) {
                0.15
            } else {
                1.0
            }
        } else if symbols.iter().any(|s| {
            s.len() >= 4 && !pkg_tokens.contains(s.as_str()) && node_name_low.contains(s.as_str())
        }) {
            0.65
        } else {
            0.0
        };
        let mut dotted_node_boost = 0.0f64;
        for (lhs, rhs) in &dotted_pairs {
            if node_qual_low == format!("{lhs}.{rhs}") {
                dotted_node_boost += 10.0;
            } else if node.kind == "class" && node_name_low == *lhs {
                dotted_node_boost += 7.0;
            } else if node_qual_low.starts_with(&format!("{lhs}.")) {
                dotted_node_boost += 2.5;
            }
        }
        let fprior = top_files
            .get(node.file.as_str())
            .map(|&rank| 1.0 / (1.0 + rank as f64))
            .unwrap_or(0.0);

        let in_repro_frame = repro.frames.iter().any(|(rf, rline, rfn)| {
            rf == &node.file
                && ((*rline >= node.start_line && *rline <= node.end_line) || rfn == &node.name)
        });
        let repro_boost = if in_repro_frame { 8.0 } else { 0.0 };

        let total = 1.0 * base
            + 4.0 * name_hit
            + 4.5 * fprior
            + 0.35 * prop
            + dotted_node_boost
            + repro_boost;
        results.push((total, id.clone()));
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut diverse = Vec::with_capacity(top_k);
    let mut per_file: HashMap<String, usize> = HashMap::new();
    for (sc, id) in results {
        let f = index
            .nodes
            .get(&id)
            .map(|n| n.file.clone())
            .unwrap_or_default();
        let cnt = per_file.entry(f).or_insert(0);
        if *cnt < 3 {
            *cnt += 1;
            diverse.push((sc, id));
            if diverse.len() >= top_k {
                break;
            }
        }
    }
    diverse
}

/// Renames the issue asks for (`` `A.old` -> `.new` ``, "rename old to new") with up to 10 source
/// lines (`file:line: text`) that define or call `old`; renames whose `old` is nowhere in the
/// sources (already done, or prose) are dropped.
pub fn find_rename_sites(
    ws: &Path,
    index: &RepoIndex,
    issue_text: &str,
) -> Vec<(String, String, Vec<String>)> {
    let mut out = Vec::new();
    for (old, new) in rename_pairs(issue_text).into_iter().take(3) {
        let site = Regex::new(&format!(
            r"(?:\.|\bdef\s+|^|[^\w.]){}\s*\(",
            regex::escape(&old)
        ))
        .expect("regex");
        let mut sites = Vec::new();
        for f in index
            .files
            .iter()
            .filter(|f| !is_test_path(f) && !is_docs_path(f))
        {
            let Ok(txt) = crate::read_lf(&crate::join_rel(ws, f)) else {
                continue;
            };
            if !txt.contains(old.as_str()) {
                continue;
            }
            for (i, line) in txt.lines().enumerate() {
                if site.is_match(line) && sites.len() < 10 {
                    sites.push(format!("{f}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        if !sites.is_empty() {
            out.push((old, new, sites));
        }
    }
    out
}

/// `(old, new)` names the issue renames: `` `A.old` -> `.new` `` or "rename old to new".
pub fn rename_pairs(issue_text: &str) -> Vec<(String, String)> {
    let arrow = Regex::new(
        r"`(?:[\w.]*\.)?(\w+)(?:\(\))?`\s*(?:->|→|=>)\s*`(?:[\w.]*\.)?\.?(\w+)(?:\(\))?`",
    )
    .expect("regex");
    let verb = Regex::new(
        r"(?i)\brenam\w*\s+`?(?:[\w.]*\.)?(\w+)(?:\(\))?`?\s+(?:to|into|as)\s+`?\.?(?:[\w.]*\.)?(\w+)",
    )
    .expect("regex");
    let mut pairs: Vec<(String, String)> = Vec::new();
    for re in [&arrow, &verb] {
        for cap in re.captures_iter(issue_text) {
            let (old, new) = (cap[1].to_string(), cap[2].to_string());
            if old.len() >= 3 && old != new && !pairs.iter().any(|(o, _)| *o == old) {
                pairs.push((old, new));
            }
        }
    }
    pairs
}

/// Five-signal multi-file `ALSO-CHANGE` counterpart detector.
/// Combines: (1) Graph neighbors, (2) Git co-change commits, (3) Rare-symbol IDF references,
/// (4) Issue identifiers, and (5) Directory + filename stem family (`tutorial001_py310`).
pub fn find_also_change_files(
    ws: &Path,
    index: &RepoIndex,
    seed_file: &str,
    seed_symbols: &[String],
    issue_text: &str,
    top_k: usize,
) -> Vec<(String, f64, usize, String)> {
    let source_files: Vec<&String> = index
        .files
        .iter()
        .filter(|f| *f != seed_file && !is_test_path(f) && !is_docs_path(f))
        .collect();
    if source_files.is_empty() {
        return Vec::new();
    }

    let mut sig_graph: HashMap<String, f64> = HashMap::new();
    let mut sig_cochg: HashMap<String, f64> = HashMap::new();
    let mut sig_refs: HashMap<String, f64> = HashMap::new();
    let mut sig_issue: HashMap<String, f64> = HashMap::new();
    let mut sig_sib: HashMap<String, f64> = HashMap::new();
    let mut reasons: HashMap<String, Vec<String>> = HashMap::new();

    // Signal 1: Code Graph neighbors (callers, callees, subclasses, adj)
    let sym_set: HashSet<&str> = seed_symbols.iter().map(|s| s.as_str()).collect();
    for (id, node) in &index.nodes {
        if node.file != seed_file {
            continue;
        }
        if !sym_set.is_empty() && !sym_set.contains(node.name.as_str()) {
            continue;
        }
        let mut linked = BTreeSet::new();
        for map in [
            &index.adj,
            &index.callers,
            &index.callees,
            &index.subclasses,
            &index.superclasses,
        ] {
            if let Some(list) = map.get(id) {
                linked.extend(list.iter().cloned());
            }
            if let Some((cls_id, _)) = id.rsplit_once('.')
                && let Some(list) = map.get(cls_id)
            {
                linked.extend(list.iter().cloned());
            }
        }
        for nb_id in linked {
            if let Some(nb_node) = index.nodes.get(&nb_id)
                && nb_node.file != seed_file
                && !is_test_path(&nb_node.file)
                && !is_docs_path(&nb_node.file)
            {
                *sig_graph.entry(nb_node.file.clone()).or_insert(0.0) += 1.0;
                let r = format!("graph: {} <-> {}", nb_node.qualname, node.qualname);
                let entry = reasons.entry(nb_node.file.clone()).or_default();
                if !entry.contains(&r) {
                    entry.push(r);
                }
            }
        }
    }

    // Export surface signal: package `__init__.py` files re-exporting symbols from `seed_file`
    let seed_path = Path::new(seed_file);
    if let Some(stem) = seed_path.file_stem().and_then(|s| s.to_str()) {
        let mut cur_dir = seed_path.parent();
        while let Some(dir) = cur_dir {
            let init_cand = if dir.as_os_str().is_empty() {
                "__init__.py".to_string()
            } else {
                crate::join_rel(dir, "__init__.py")
                    .to_string_lossy()
                    .replace('\\', "/")
            };
            if init_cand != seed_file
                && index.files.contains(&init_cand)
                && let Ok(src) = crate::read_lf(&crate::join_rel(ws, &init_cand))
                && (src.contains(stem) || seed_symbols.iter().any(|s| src.contains(s.as_str())))
            {
                *sig_graph.entry(init_cand.clone()).or_insert(0.0) += 1.5;
                let r = format!("export: {init_cand} re-exports {stem}");
                let entry = reasons.entry(init_cand.clone()).or_default();
                if !entry.contains(&r) {
                    entry.push(r);
                }
            }
            cur_dir = dir.parent();
        }
    }

    // Signal 2: Git co-change history (`git log -n 1000 --name-only --format=%x00`)
    if let Ok(out) = Command::new("git")
        .args([
            "log",
            "-n",
            "1000",
            "--name-only",
            "--format=%x00",
            "--",
            ".",
        ])
        .current_dir(ws)
        .output()
        && out.status.success()
    {
        let log_txt = String::from_utf8_lossy(&out.stdout);
        let commits: Vec<HashSet<&str>> = log_txt
            .split('\0')
            .map(|blk| {
                blk.lines()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect::<HashSet<_>>()
            })
            .filter(|c| !c.is_empty() && c.len() <= 40)
            .collect();
        let with_seed: Vec<&HashSet<&str>> =
            commits.iter().filter(|c| c.contains(seed_file)).collect();
        if !with_seed.is_empty() {
            let denom = with_seed.len() as f64;
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for c in &with_seed {
                for &f in *c {
                    if f != seed_file {
                        *counts.entry(f).or_insert(0) += 1;
                    }
                }
            }
            for (f, n) in counts {
                if n >= 2 && !is_test_path(f) && !is_docs_path(f) && ws.join(f).is_file() {
                    sig_cochg.insert(f.to_string(), (n as f64) / denom);
                    reasons.entry(f.to_string()).or_default().push(format!(
                        "history: co-changed in {n}/{} commits",
                        with_seed.len()
                    ));
                }
            }
        }
    }

    // Read source texts for Signals 3 & 4
    let mut texts: HashMap<&str, String> = HashMap::new();
    for f in &source_files {
        if let Ok(t) = std::fs::read_to_string(ws.join(f.as_str())) {
            texts.insert(f.as_str(), t);
        }
    }
    let n_files = (texts.len() + 1) as f64;

    // Signal 3: Rare-symbol IDF references
    let mut df: HashMap<&str, usize> = HashMap::new();
    for sym in seed_symbols {
        if sym.len() <= 2 {
            continue;
        }
        let c = texts.values().filter(|t| t.contains(sym.as_str())).count();
        if c > 0 {
            df.insert(sym.as_str(), c);
        }
    }
    for (&f, txt) in &texts {
        let mut v = 0.0f64;
        let mut used = Vec::new();
        for (&sym, &c) in &df {
            if txt.contains(sym) {
                v += (n_files / (c as f64)).ln().max(0.0);
                used.push(sym);
            }
        }
        if v > 0.0 {
            sig_refs.insert(f.to_string(), v);
            used.sort_unstable();
            reasons.entry(f.to_string()).or_default().push(format!(
                "refs: uses {}",
                used.into_iter().take(3).collect::<Vec<_>>().join(", ")
            ));
        }
    }

    // Signal 4: Issue identifiers present in candidate file
    let (_, issue_syms, _) = extract_query_terms(issue_text);
    if !issue_syms.is_empty() {
        let denom = issue_syms.len() as f64;
        for (&f, txt) in &texts {
            let hits = issue_syms
                .iter()
                .filter(|w| txt.contains(w.as_str()))
                .count();
            if hits > 0 {
                sig_issue.insert(f.to_string(), (hits as f64) / denom);
            }
        }
    }

    // Signal 5: Same directory + stem family (`tutorial001` <-> `tutorial001_py310`)
    let seed_fam = stem_family(seed_file);
    for f in &source_files {
        if stem_family(f) == seed_fam {
            sig_sib.insert((*f).clone(), 1.0);
            reasons
                .entry((*f).clone())
                .or_default()
                .push(format!("sibling stem of {seed_file}"));
        }
    }

    // Combine normalized signals + agreement bonus
    let weights: [(&HashMap<String, f64>, f64); 5] = [
        (&sig_graph, 1.0),
        (&sig_cochg, 1.5),
        (&sig_refs, 1.0),
        (&sig_issue, 0.5),
        (&sig_sib, 1.0),
    ];
    let mut combined: HashMap<String, (f64, usize)> = HashMap::new();
    for (sig_map, w) in weights {
        let max_v = sig_map.values().copied().fold(0.0f64, f64::max);
        if max_v <= 0.0 {
            continue;
        }
        for (f, &v) in sig_map {
            let entry = combined.entry(f.clone()).or_insert((0.0, 0));
            entry.0 += w * (v / max_v);
            entry.1 += 1;
        }
    }

    let mut out: Vec<(String, f64, usize, String)> = combined
        .into_iter()
        .map(|(f, (s, agree))| {
            let score = ((s + 0.5 * ((agree.saturating_sub(1)) as f64)) * 100.0).round() / 100.0;
            let why = reasons
                .remove(&f)
                .unwrap_or_default()
                .into_iter()
                .take(2)
                .collect::<Vec<_>>()
                .join("; ");
            (f, score, agree, why)
        })
        .filter(|(_, score, agree, _)| *score >= 1.1 || *agree >= 2)
        .collect();

    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_k);
    out
}

fn stem_family(path: &str) -> (String, String) {
    let p = Path::new(path);
    let dir = p
        .parent()
        .map(|d| d.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let re = Regex::new(r"_(py\d+|an|py\d+_an|an_py\d+)$").expect("regex");
    let clean = re.replace(&stem, "").into_owned();
    (dir, clean)
}

/// Find paired test files for the top target definitions (`tests_of` graph edges + stem matching).
pub fn find_paired_tests(index: &RepoIndex, top_defs: &[&DefNode]) -> Vec<String> {
    let mut tests = Vec::new();
    for d in top_defs {
        if let Some(ts) = index.tests_of.get(&d.id) {
            for t in ts {
                if !tests.contains(t) {
                    tests.push(t.clone());
                }
            }
        }
    }
    for d in top_defs {
        let stem = Path::new(&d.file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if stem.is_empty() || stem == "__init__" {
            continue;
        }
        for f in &index.files {
            if is_test_path(f)
                && (f.contains(&format!("test_{stem}")) || f.contains(&format!("{stem}_test")))
                && !tests.contains(f)
            {
                tests.push(f.clone());
            }
        }
    }
    tests.into_iter().take(4).collect()
}

/// Execute `/tmp/coder debug` and return the formatted Debug Brief + Preloaded Code.
pub fn run_debug(env: &EnvPaths, args: &BTreeMap<String, String>, body: &str) -> Result<String> {
    let query = args
        .get("query")
        .or_else(|| args.get("issue"))
        .or_else(|| args.get("q"))
        .map(|s| s.as_str())
        .unwrap_or(body)
        .trim();

    let explicit_repro = args
        .get("repro")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("");

    // Persist issue/query text for later contract checks in `patch` / `verify`
    let issue_cache = env.agent_dir.join("issue.txt");
    // The engine sees the task's problem statement only when the model passes it (`issue=`);
    // otherwise every issue-based check (renames, exact literals, removal intent, ALSO-CHANGE
    // issue identifiers) works from the model's query.
    let issue_arg = args.get("issue").map(|s| s.trim()).unwrap_or("");
    let base = if issue_arg.is_empty() {
        query.to_string()
    } else {
        format!("{issue_arg}\n{query}")
    };
    let combined_issue = if !explicit_repro.is_empty() {
        format!("{base}\n```python\n{explicit_repro}\n```")
    } else {
        base
    };
    if !combined_issue.trim().is_empty()
        && (!issue_cache.is_file() || args.contains_key("issue") || !explicit_repro.is_empty())
    {
        if !explicit_repro.is_empty() {
            let _ = std::fs::remove_file(env.agent_dir.join("repro_meta.json"));
        }
        let _ = crate::write_lf(&issue_cache, &combined_issue);
    }
    let full_issue = crate::read_lf(&issue_cache).unwrap_or(combined_issue);
    let search_query_owned = if !explicit_repro.is_empty() && !query.is_empty() {
        format!("{query}\n{explicit_repro}")
    } else if query.is_empty() {
        full_issue.clone()
    } else {
        query.to_string()
    };
    // Code identifiers the issue puts in backticks are the strongest localization signal, even
    // when the model's own query paraphrases them: add them to the search query.
    let search_query_owned = if issue_arg.is_empty() {
        search_query_owned
    } else {
        let mut extra: Vec<&str> = Vec::new();
        for cap in Regex::new(r"`([A-Za-z_][\w.]{2,60})(?:\(\))?`")
            .expect("regex")
            .captures_iter(issue_arg)
        {
            let ident = cap.get(1).map_or("", |m| m.as_str());
            if !search_query_owned.contains(ident) && !extra.contains(&ident) && extra.len() < 8 {
                extra.push(ident);
            }
        }
        if extra.is_empty() {
            search_query_owned
        } else {
            format!("{search_query_owned} {}", extra.join(" "))
        }
    };
    let search_query = search_query_owned.as_str();

    let index = load_or_build_index(env)?;
    let repro = run_or_load_repro(env, &full_issue, &index);
    let file_rank = rank_files_dm1(&env.ws, &index, search_query, &repro);
    let fn_rank = diversify_by_file(
        &index,
        rank_functions_graph(&index, search_query, &file_rank, &repro, 18),
        6,
    );

    let mut out = String::new();

    // 1. REPRO Status
    if repro.has_repro && repro.fails_now {
        out.push_str(&format!("REPRO: FAILS NOW ({})\n", repro.summary));
        if !repro.frames.is_empty() {
            let frames_str: Vec<String> = repro
                .frames
                .iter()
                .map(|(f, l, fnm)| format!("{f}:{l} in {fnm}"))
                .collect();
            out.push_str(&format!(
                "  TRACEBACK FRAMES: {}\n",
                frames_str.join(" -> ")
            ));
        }
    } else if !repro.frames.is_empty() {
        let frames_str: Vec<String> = repro
            .frames
            .iter()
            .map(|(f, l, fnm)| format!("{f}:{l} in {fnm}"))
            .collect();
        out.push_str(&format!(
            "REPRO: issue traceback frames -> {}\n",
            frames_str.join(" -> ")
        ));
    } else {
        out.push_str("REPRO: none yet — after patching, call `run_skill_script` (`args={\"repro\": \"import ...\\nassert ...\"}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"`)!\n");
    }

    // 2. TOP TARGETS (Code Graph + BM25 + 1-Hop Propagation)
    let top_nodes: Vec<&DefNode> = fn_rank
        .iter()
        .filter_map(|(_, id)| index.nodes.get(id))
        .collect();

    out.push_str("TOP TARGETS (AST + Call/Inheritance Graph + BM25):\n");
    if top_nodes.is_empty() {
        for (i, (_, f)) in file_rank
            .iter()
            .filter(|(_, f)| !is_test_path(f))
            .take(4)
            .enumerate()
        {
            out.push_str(&format!("  {}. {}\n", i + 1, f));
        }
    } else {
        for (i, d) in top_nodes.iter().take(5).enumerate() {
            let callers_preview: Vec<String> = index
                .callers
                .get(&d.id)
                .into_iter()
                .flatten()
                .filter_map(|cid| index.nodes.get(cid))
                .take(2)
                .map(|cn| format!("{}:{}", cn.file, cn.qualname))
                .collect();
            let callers_tag = if callers_preview.is_empty() {
                String::new()
            } else {
                format!(" | callers: {}", callers_preview.join(", "))
            };
            out.push_str(&format!(
                "  {}. {}:{}..{} ({} {}){}\n",
                i + 1,
                d.file,
                d.start_line,
                d.end_line,
                if d.kind == "async_function" {
                    "async def"
                } else if d.kind == "class" {
                    "class"
                } else {
                    "def"
                },
                d.qualname,
                callers_tag
            ));
        }
    }

    let top_file_list: Vec<String> = file_rank
        .iter()
        .filter(|(_, f)| !is_test_path(f) && !is_docs_path(f))
        .take(5)
        .map(|(sc, f)| format!("{f} ({sc:.1})"))
        .collect();
    if !top_file_list.is_empty() {
        out.push_str(&format!("TOP FILES: {}\n", top_file_list.join(", ")));
    }

    // 3. 5-Signal ALSO-CHANGE Multi-File Counterparts
    let seed_file = top_nodes
        .first()
        .map(|d| d.file.as_str())
        .or_else(|| {
            file_rank
                .iter()
                .find(|(_, f)| !is_test_path(f) && !is_docs_path(f))
                .map(|(_, f)| f.as_str())
        })
        .unwrap_or("");
    let seed_syms: Vec<String> = top_nodes
        .iter()
        .filter(|d| d.file == seed_file)
        .map(|d| d.name.clone())
        .collect();
    let also_change = if !seed_file.is_empty() {
        find_also_change_files(&env.ws, &index, seed_file, &seed_syms, &full_issue, 3)
    } else {
        Vec::new()
    };

    if !also_change.is_empty() {
        out.push_str("ALSO-CHANGE (multi-file counterparts to check/patch together):\n");
        for (f, score, agree, why) in &also_change {
            out.push_str(&format!(
                "  - {f} (score={score:.2}, signals={agree}/5: {why})\n"
            ));
        }
    }

    // 3b. Renames named in the issue ("`HTTPParser.complete` -> `.reset`"): every source call site
    // must follow the definition
    for (old, new, sites) in find_rename_sites(&env.ws, &index, &full_issue) {
        out.push_str(&format!(
            "RENAME `{old}` -> `{new}`: rename the def AND update every call site below (tests are not yours to edit):\n"
        ));
        for s in &sites {
            out.push_str(&format!("  - {s}\n"));
        }
    }

    // 4. Whitened Structural Embedding Sibling
    if let Some(primary) = top_nodes.first()
        && let Some((sib_id, sim)) = find_structural_sibling(&index, &primary.id)
        && let Some(sib_node) = index.nodes.get(&sib_id)
    {
        out.push_str(&format!(
            "SIBLING PATTERN (whitened structural embedding match): {}:{}..{} ({}, cos={:.2})\n",
            sib_node.file, sib_node.start_line, sib_node.end_line, sib_node.qualname, sim
        ));
    }

    // 5. Paired Test Files
    let paired_tests = find_paired_tests(&index, &top_nodes);
    if !paired_tests.is_empty() {
        out.push_str(&format!("PAIRED TESTS: {}\n", paired_tests.join(", ")));
    }

    // 6. Auto-Preload Source Code of Top 2 Primary Targets (+ Class Method Outline for large classes) + Top 1 ALSO-CHANGE Counterpart
    out.push_str("\nPRELOADED CODE (pass 2-8 exact lines in `old_string` + `new_string` to `scripts/patch.py`):\n");
    let mut preloaded_spans: Vec<(String, usize, usize, String)> = Vec::new();
    for d in top_nodes.iter().take(2) {
        let span_lines = d.end_line.saturating_sub(d.start_line) + 1;
        if d.kind == "class" && span_lines > 85 {
            // Emit the method outline of a large class instead of its first lines
            if let Some(fdefs) = index.file_defs.get(&d.file) {
                let prefix = format!("{}.", d.qualname);
                let methods = crate::view::unique_class_methods(
                    fdefs
                        .iter()
                        .filter_map(|id| index.nodes.get(id))
                        .filter(|m| m.qualname.starts_with(&prefix)),
                    110,
                );
                if !methods.is_empty() {
                    out.push_str(&format!(
                        "METHODS OF class `{}` in `{}`: {}\n",
                        d.qualname,
                        d.file,
                        methods.join(", ")
                    ));
                }
            }
            let body_s = if d.body_start_line > d.start_line + 25 {
                d.body_start_line.saturating_sub(2)
            } else {
                d.start_line
            };
            let end_cap = d.end_line.min(body_s + 40);
            preloaded_spans.push((
                d.file.clone(),
                body_s,
                end_cap,
                format!("{} {} body", d.kind, d.qualname),
            ));
            // Also preload the highest-ranked method belonging to this class if not already in top 2
            let prefix = format!("{}.", d.qualname);
            if let Some(best_meth) = top_nodes
                .iter()
                .skip(2)
                .find(|m| m.file == d.file && m.qualname.starts_with(&prefix))
            {
                let ms = if best_meth.body_start_line > best_meth.start_line + 25 {
                    best_meth.body_start_line.saturating_sub(2)
                } else {
                    best_meth.start_line
                };
                let mend = best_meth.end_line.min(ms + 75);
                preloaded_spans.push((
                    best_meth.file.clone(),
                    ms,
                    mend,
                    format!("{} {}", best_meth.kind, best_meth.qualname),
                ));
            }
        } else {
            let bs = if d.body_start_line > d.start_line + 25 {
                d.body_start_line.saturating_sub(2)
            } else {
                d.start_line
            };
            let end_cap = d.end_line.min(bs + 85);
            preloaded_spans.push((
                d.file.clone(),
                bs,
                end_cap,
                format!("{} {}", d.kind, d.qualname),
            ));
        }
    }
    // If there is a strong ALSO-CHANGE file not yet in preloaded_spans, preload its top symbol
    if let Some((counterpart_file, _, _, _)) = also_change.first()
        && !preloaded_spans
            .iter()
            .any(|(f, _, _, _)| f == counterpart_file)
        && let Some(c_defs) = index.file_defs.get(counterpart_file)
    {
        let best_c = c_defs
            .iter()
            .filter_map(|id| index.nodes.get(id))
            .find(|d| {
                seed_syms
                    .iter()
                    .any(|s| d.calls.iter().any(|(_, c)| c == s))
            })
            .or_else(|| c_defs.first().and_then(|id| index.nodes.get(id)));
        if let Some(cd) = best_c {
            let end_cap = cd.end_line.min(cd.start_line + 55);
            preloaded_spans.push((
                cd.file.clone(),
                cd.start_line,
                end_cap,
                format!("counterpart {} {}", cd.kind, cd.qualname),
            ));
        }
    }

    let mut recent_targets: Vec<String> = preloaded_spans
        .iter()
        .map(|(f, _, _, _)| f.clone())
        .collect();
    for (_, f) in file_rank
        .iter()
        .filter(|(_, f)| !is_test_path(f) && !is_docs_path(f))
        .take(4)
    {
        if !recent_targets.contains(f) {
            recent_targets.push(f.clone());
        }
    }
    crate::record_recent_files(env, &recent_targets);

    for (file, start, end, label) in &preloaded_spans {
        if let Ok(content) = crate::read_lf(&crate::join_rel(&env.ws, file)) {
            out.push_str(&format!("--- {file}:{start}-{end} ({label}) ---\n"));
            out.push_str(&format_numbered_slice(&content, *start, *end));
        }
    }

    out.push_str(
        "\nNEXT: every `run_skill_script` call is `{\"args\": {...}, \"file_path\": \"scripts/<name>.py\", \"skill_name\": \"swe-coder\"}` (ALL parameters inside `args`; after `args` closes, ALWAYS add `file_path` and `skill_name`). Inspect via `file_path=\"scripts/view.py\"` (`args={\"file\": \"...\", \"symbol\": \"Class.method\"}` or `\"symbol\": \"imports\"`), edit via `file_path=\"scripts/patch.py\"` (`args={\"file\": \"...\", \"symbol\": \"Class.method\", \"old_string\": \"...\", \"new_string\": \"...\"}`), then verify via `file_path=\"scripts/verify.py\"` (`args={\"repro\": \"import ...\\nassert ...\"}`).\n",
    );

    Ok(clip_tokens(&out, 1800))
}
