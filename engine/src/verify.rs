//! `/tmp/coder verify` (reproducer + graph-paired pytest + static critic + hygiene gate, in parallel)
//! and `/tmp/coder probe` (15 s Python snippet runner that keeps `/workspace` clean).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use regex::Regex;

use crate::ast_graph::{check_python_syntax, is_test_path, tokenize_ident_and_subwords};
use crate::patch::{MergeLockGuard, run_static_critic, save_best_patch_snapshot};
use crate::{EnvPaths, clip_tokens, load_or_build_index, run_shell_with_timeout};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PytestStatus {
    Pass,
    Fail,
    EnvSkip,
    NoTestsFound,
}

/// Strip markdown code fences if a model accidentally wrapped an inline Python snippet.
fn strip_code_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut lines: Vec<&str> = trimmed.lines().collect();
    if !lines.is_empty() && lines[0].trim_start().starts_with("```") {
        lines.remove(0);
    }
    if !lines.is_empty() && lines.last().is_some_and(|l| l.trim().starts_with("```")) {
        lines.pop();
    }
    lines.join("\n")
}

/// Attempt to repair a uniform indentation shift on the modified hunk of `broken_src` relative to `HEAD:<rel_file>`.
pub fn repair_modified_hunk_indentation(
    ws: &Path,
    rel_file: &str,
    broken_src: &str,
) -> Option<String> {
    let old_src = crate::patch::git_show_head(ws, rel_file);
    if old_src.is_empty() {
        return None;
    }
    let old_lines: Vec<&str> = old_src.lines().collect();
    let new_lines: Vec<&str> = broken_src.lines().collect();
    let mut pref = 0usize;
    while pref < old_lines.len() && pref < new_lines.len() && old_lines[pref] == new_lines[pref] {
        pref += 1;
    }
    let mut suff = 0usize;
    while suff < (old_lines.len().saturating_sub(pref))
        && suff < (new_lines.len().saturating_sub(pref))
        && old_lines[old_lines.len() - 1 - suff] == new_lines[new_lines.len() - 1 - suff]
    {
        suff += 1;
    }
    let hunk_end = new_lines.len().saturating_sub(suff);
    if pref >= hunk_end {
        return None;
    }
    let trailing_nl = broken_src.ends_with('\n');
    for delta in [-4isize, 4, -8, 8, -2, 2, -12, 12] {
        let mut shifted: Vec<String> = Vec::with_capacity(new_lines.len());
        for &ln in &new_lines[..pref] {
            shifted.push(ln.to_string());
        }
        let mut ok = true;
        for &ln in &new_lines[pref..hunk_end] {
            if ln.trim().is_empty() {
                shifted.push(String::new());
                continue;
            }
            let lead = ln.bytes().take_while(|&b| b == b' ').count();
            let new_lead = lead as isize + delta;
            if new_lead < 0 {
                ok = false;
                break;
            }
            shifted.push(format!("{}{}", " ".repeat(new_lead as usize), &ln[lead..]));
        }
        if !ok {
            continue;
        }
        for &ln in &new_lines[hunk_end..] {
            shifted.push(ln.to_string());
        }
        let mut cand = shifted.join("\n");
        if trailing_nl {
            cand.push('\n');
        }
        if check_python_syntax(&cand).is_none()
            && crate::ast_graph::find_misindented_nested_methods(&cand).is_empty()
        {
            return Some(cand);
        }
    }
    None
}

/// Ensure `/workspace` has clean hygiene (no scratch files, no modified tests/configs, no destructive wipe)
/// and restore `/tmp/.agent/best.patch` if `/workspace` diff is empty.
pub fn ensure_clean_submittable_workspace(env: &EnvPaths) -> (Vec<String>, bool, String) {
    let hygiene_actions = enforce_workspace_hygiene(env);
    let mut restored_backup = false;
    let mut diff_text = git_diff_head(&env.ws);
    if diff_text.trim().is_empty() {
        let best_patch = env.agent_dir.join("best.patch");
        if best_patch.is_file()
            && let Ok(bytes) = std::fs::read(&best_patch)
            && !bytes.is_empty()
        {
            let status = Command::new("git")
                .args(["apply", "--whitespace=nowarn"])
                .arg(&best_patch)
                .current_dir(&env.ws)
                .status();
            if status.map(|s| s.success()).unwrap_or(false) {
                restored_backup = true;
                diff_text = git_diff_head(&env.ws);
            }
        }
    }
    (hygiene_actions, restored_backup, diff_text)
}

/// Run `/tmp/coder verify`:
/// 1. Clean untracked scratch files & revert accidental test/config modifications.
/// 2. Restore `/tmp/.agent/best.patch` if `/workspace` diff was accidentally wiped.
/// 3. An inline `repro` argument or heredoc body is saved as `/tmp/.agent/repro.py`.
/// 4. Run in parallel (`std::thread::scope`):
///    - Worker A: `/tmp/.agent/repro.py` (if captured or passed via `repro=`)
///    - Worker B: Graph-paired `pytest` (`tests_of` + stem/token match + missing-plugin fallback)
///    - Worker C: Static Critic (`blocking` + `advisory`) + Issue literal check
pub fn run_verify(
    env: &EnvPaths,
    args: &BTreeMap<String, String>,
    body: &str,
    elapsed_secs: u64,
) -> Result<String> {
    let _guard = MergeLockGuard::acquire(&env.agent_dir);

    // Save inline `repro` snippet if provided in `args` or `body`
    let raw_repro = args
        .get("repro")
        .or_else(|| args.get("reproducer"))
        .map(String::as_str)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| (!body.trim().is_empty()).then_some(body));
    if let Some(rp_raw) = raw_repro {
        let mut rp_code = strip_code_fences(rp_raw);
        if check_python_syntax(&rp_code).is_some() {
            let unesc = crate::patch::unescape_once(&rp_code);
            if check_python_syntax(&unesc).is_none() {
                rp_code = unesc;
            }
        }
        if let Some((line, syn_err)) = check_python_syntax(&rp_code) {
            return Ok(format!(
                "VERDICT: REPRODUCER SYNTAX ERROR (line {line}): {syn_err}\nNEXT: Re-run `scripts/verify.py` with valid Python syntax in `args={{\"repro\": \"import ...\\nassert ...\"}}`."
            ));
        }
        let _ = crate::write_lf(&env.agent_dir.join("repro.py"), &rp_code);
    }

    let (mut hygiene_actions, restored_backup, mut diff_text) =
        ensure_clean_submittable_workspace(env);

    if diff_text.trim().is_empty() {
        return Ok(
            "VERDICT: EMPTY DIFF — `/workspace` has no changes (`git diff HEAD` is 0 bytes).\nNEXT: Apply your fix via `scripts/patch.py` (`args={\"file\": \"...\", \"symbol\": \"...\", \"old_string\": \"...\", \"new_string\": \"...\"}`) before calling `submit_patch`."
                .to_string(),
        );
    }

    let modified_files = git_modified_files(&env.ws);
    if !modified_files.is_empty() {
        let _ = std::fs::remove_file(env.agent_dir.join("index.json"));
    }

    // Syntax gate across all modified `.py` files (with auto-unescape + uniform hunk indentation repair)
    let mut syntax_errors = Vec::new();
    let mut repaired_any = false;
    for f in &modified_files {
        if f.ends_with(".py") {
            let full_p = crate::join_rel(&env.ws, f);
            if let Ok(src) = crate::read_lf(&full_p) {
                if let Some((line, err)) = check_python_syntax(&src) {
                    let unesc = crate::patch::unescape_once(&src);
                    if unesc != src && check_python_syntax(&unesc).is_none() {
                        let _ = crate::write_lf(&full_p, &unesc);
                        hygiene_actions.push(format!("auto-unescaped literal \\\"/\\n in `{f}`"));
                        repaired_any = true;
                    } else if let Some(indented) =
                        repair_modified_hunk_indentation(&env.ws, f, &src)
                    {
                        let _ = crate::write_lf(&full_p, &indented);
                        hygiene_actions
                            .push(format!("auto-repaired modified hunk indentation in `{f}`"));
                        repaired_any = true;
                    } else {
                        syntax_errors.push(format!("{f}:{line}: {err}"));
                    }
                } else if !crate::ast_graph::find_misindented_nested_methods(&src).is_empty()
                    && crate::ast_graph::find_misindented_nested_methods(
                        &crate::patch::git_show_head(&env.ws, f),
                    )
                    .is_empty()
                    && let Some(indented) = repair_modified_hunk_indentation(&env.ws, f, &src)
                {
                    let _ = crate::write_lf(&full_p, &indented);
                    hygiene_actions
                        .push(format!("auto-repaired modified hunk indentation in `{f}`"));
                    repaired_any = true;
                }
            }
        }
    }
    if repaired_any {
        diff_text = git_diff_head(&env.ws);
    }
    if !syntax_errors.is_empty() {
        // Near timeout, fall back to last clean `best.patch` rather than leaving a broken workspace
        let best_patch = env.agent_dir.join("best.patch");
        let syntax_fallback_secs = (190.0 * crate::budget_scale()) as u64;
        if elapsed_secs >= syntax_fallback_secs && best_patch.is_file() {
            let _ = Command::new("git")
                .args(["checkout", "HEAD", "--", "."])
                .current_dir(&env.ws)
                .output();
            if Command::new("git")
                .args(["apply", "--whitespace=nowarn"])
                .arg(&best_patch)
                .current_dir(&env.ws)
                .status()
                .is_ok_and(|s| s.success())
            {
                return Ok(
                    "HYGIENE: Reverted late syntax error and restored last clean `best.patch`.\nVERDICT: READY TO CALL `submit_patch` NOW."
                        .to_string(),
                );
            }
        }
        return Ok(format!(
            "VERDICT: SYNTAX ERROR IN WORKSPACE:\n  - {}\nNEXT: Fix syntax/indentation via `scripts/patch.py` (inspect around the reported line with `scripts/view.py` first).",
            syntax_errors.join("\n  - ")
        ));
    }

    // Save clean snapshot to `/tmp/.agent/best.patch`
    let diff_bytes = save_best_patch_snapshot(env);

    let explicit_test = args
        .get("test")
        .or_else(|| args.get("tests"))
        .cloned()
        .unwrap_or_default();

    let issue_text = crate::read_lf(&env.agent_dir.join("issue.txt")).unwrap_or_default();

    // Pre-build or load the repo index before spawning threads so Workers B & C share it without racing
    let index_opt = load_or_build_index(env).ok();
    let test_targets = select_test_targets(
        &env.ws,
        index_opt.as_ref(),
        &modified_files,
        &explicit_test,
        &issue_text,
    );

    // Step 4: Run Repro, Pytest, and Static Critic in parallel across CPU threads
    let (
        (repro_status, repro_detail),
        (pytest_status, pytest_detail, ran_targets),
        (critic_report, literal_notes),
    ) = std::thread::scope(|s| {
        let h_repro = s.spawn(|| run_repro_worker(env));
        let h_pytest = s.spawn(|| run_pytest_worker(env, &test_targets, elapsed_secs));
        let h_critic = s.spawn(|| {
            let critic = run_static_critic(env, &modified_files);
            let lit = check_issue_literals(env, &diff_text);
            (critic, lit)
        });

        (
            h_repro.join().unwrap_or((None, String::new())),
            h_pytest.join().unwrap_or((
                PytestStatus::NoTestsFound,
                String::new(),
                test_targets.clone(),
            )),
            h_critic.join().unwrap_or_default(),
        )
    });

    let mut out = String::new();
    out.push_str(&format!(
        "DIFF SUMMARY: {} file(s) modified ({}) — {} bytes (saved to `best.patch`)\n",
        modified_files.len(),
        modified_files.join(", "),
        diff_bytes
    ));
    if restored_backup {
        out.push_str(
            "HYGIENE: Restored `/tmp/.agent/best.patch` because `/workspace` diff was empty.\n",
        );
    }
    if !hygiene_actions.is_empty() {
        out.push_str(&format!("HYGIENE: {}\n", hygiene_actions.join("; ")));
    }

    // Repro report
    match repro_status {
        Some(true) if repro_detail.is_empty() => out.push_str("REPRODUCER: PASS (exits 0)\n"),
        Some(true) => out.push_str(&format!("REPRODUCER: PASS (exits 0)\n{repro_detail}\n")),
        Some(false) => out.push_str(&format!("REPRODUCER: FAIL\n{repro_detail}\n")),
        None => out.push_str("REPRODUCER: none captured\n"),
    }

    // Pytest report
    match pytest_status {
        PytestStatus::Pass => {
            out.push_str(&format!(
                "PYTEST ({}): PASS\n{}\n",
                ran_targets.join(", "),
                pytest_detail
            ));
        }
        PytestStatus::Fail => {
            out.push_str(&format!(
                "PYTEST ({}): FAIL\n{}\n",
                ran_targets.join(", "),
                pytest_detail
            ));
        }
        PytestStatus::EnvSkip if pytest_detail.starts_with("PRE-EXISTING") => {
            out.push_str(&format!(
                "PYTEST ({}): no NEW failures\n{}\n",
                ran_targets.join(", "),
                pytest_detail
            ));
        }
        PytestStatus::EnvSkip => {
            out.push_str(&format!(
                "PYTEST ({}): ENV_SKIP (missing container test plugin / C extension; rely on REPRODUCER + STATIC CRITIC)\n{}\n",
                ran_targets.join(", "),
                pytest_detail
            ));
        }
        PytestStatus::NoTestsFound => {
            out.push_str("PYTEST: no paired test file found\n");
        }
    }

    // Critic report (`blocking` errors block READY; `advisory` notes are informational)
    if critic_report.is_empty() {
        out.push_str("STATIC CRITIC: CLEAN\n");
    } else {
        if !critic_report.blocking.is_empty() {
            out.push_str(&format!(
                "STATIC CRITIC ERRORS (MUST FIX):\n  - {}\n",
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
    if !literal_notes.is_empty() {
        out.push_str(&format!("ISSUE AUDIT: {}\n", literal_notes.join("; ")));
    }

    // Overall Verdict
    let repro_failed = matches!(repro_status, Some(false));
    let pytest_failed = pytest_status == PytestStatus::Fail;
    let blocking_critic = !critic_report.blocking.is_empty();

    let repro_gate_secs = (150.0 * crate::budget_scale()) as u64;
    if repro_failed || pytest_failed || blocking_critic {
        let mut counterpart_hint = String::new();
        if modified_files.len() == 1 {
            let recent = crate::load_recent_files(env);
            if let Some(c) = recent.iter().find(|f| !modified_files.contains(f)) {
                counterpart_hint = format!(
                    " If your fix in `{}` is complete, the failure may require updating counterpart `{c}`.",
                    modified_files[0]
                );
            }
        }
        out.push_str(&format!(
            "\nVERDICT: NEEDS FIX — the patch fails the check above, and a failing patch scores 0. Read the failure, then revise via `run_skill_script` (`file_path=\"scripts/patch.py\"`, `skill_name=\"swe-coder\"`) or view the code the traceback names first.{counterpart_hint} Then re-run `file_path=\"scripts/verify.py\"`."
        ));
    } else if repro_detail.contains("WARNING: this repro only calls private names")
        && elapsed_secs <= (220.0 * crate::budget_scale()) as u64
    {
        out.push_str(
            "\nVERDICT: NEEDS ISSUE REPRO — your repro checks your own edit, not the issue. Re-run `scripts/verify.py` with a repro that calls the PUBLIC API from the issue (the call a user makes) and ends in `assert <result the issue expects>`.",
        );
    } else if repro_detail.contains("WARNING: this repro exits 0 without checking")
        && elapsed_secs <= (220.0 * crate::budget_scale()) as u64
    {
        // A repro that asserts nothing cannot show the issue is fixed: repeat the demand for a
        // checking repro while time remains.
        out.push_str(
            "\nVERDICT: NEEDS ISSUE REPRO — your repro asserts nothing, so it cannot show the issue is fixed. Re-run `scripts/verify.py` with `args={\"repro\": \"from pkg.mod import fn\\nassert fn(...) == <expected value from the issue>\"}` (straight-line code, no try/except) before calling `submit_patch`.",
        );
    } else if (repro_status.is_none() || (repro_status == Some(true) && !repro_detail.is_empty()))
        && env.agent_dir.join("issue.txt").is_file()
        && !env.agent_dir.join("repro_prompted.flag").is_file()
        && elapsed_secs <= repro_gate_secs
    {
        let _ = crate::write_lf(&env.agent_dir.join("repro_prompted.flag"), "1\n");
        out.push_str(
            "\nVERDICT: NEEDS ISSUE REPRO — existing tests pass, but no issue-specific reproducer ran yet! Call `run_skill_script` (`args={\"repro\": \"import ...\\nassert ...\"}`, `file_path=\"scripts/verify.py\"`, `skill_name=\"swe-coder\"`) ONCE before calling `submit_patch`.",
        );
    } else {
        if pytest_status == PytestStatus::EnvSkip && !pytest_detail.starts_with("PRE-EXISTING") {
            // When the existing tests could not run, the repro is the only check: say so in the reply.
            out.push_str("\nCAUTION: the existing tests could not run here, so your repro is the ONLY check. It must call the public API from the issue and assert the exact expected value; if it does not, fix the repro before submitting.");
        }
        out.push_str("\nVERDICT: READY TO CALL `submit_patch` NOW.");
    }

    Ok(clip_tokens(&out, 650))
}

/// Run `/tmp/coder probe`: execute an inline Python snippet inside `/tmp/.agent/probe/` (`PYTHONPATH=/workspace`)
/// with a 15s timeout and automatic cleanup of any stray files created in `/workspace`.
pub fn run_probe(env: &EnvPaths, args: &BTreeMap<String, String>, body: &str) -> Result<String> {
    let mut code = body.trim().to_string();
    if code.is_empty() {
        if let Some(c) = args.get("code").or_else(|| args.get("script")) {
            code = c.trim().to_string();
        } else if let Some(cmd) = args.get("cmd").or_else(|| args.get("command")) {
            let probe_dir = env.agent_dir.join("probe");
            let _ = std::fs::create_dir_all(&probe_dir);
            let before_untracked = git_untracked_files(&env.ws);
            let (rc, out) = run_shell_with_timeout(cmd, &probe_dir, &env.ws, 15);
            clean_new_untracked(&env.ws, &before_untracked);
            let summary = format!("PROBE CMD (exit {rc}):\n{}", tail_lines(&out, 25));
            return Ok(clip_tokens(&summary, 400));
        }
    }

    if code.is_empty() {
        return Ok("USAGE: /tmp/coder probe <<'EOF'\nimport pkg\n...\nEOF".to_string());
    }

    // Strip accidental ```python fences if the model wrapped the heredoc body
    if code.starts_with("```") {
        let mut lines: Vec<&str> = code.lines().collect();
        if !lines.is_empty() && lines[0].trim_start().starts_with("```") {
            lines.remove(0);
        }
        if !lines.is_empty() && lines.last().unwrap().trim().starts_with("```") {
            lines.pop();
        }
        code = lines.join("\n");
    }

    if let Some((line, syn_err)) = check_python_syntax(&code) {
        return Ok(format!("PROBE SYNTAX ERROR (line {line}): {syn_err}"));
    }

    let probe_dir = env.agent_dir.join("probe");
    let _ = std::fs::create_dir_all(&probe_dir);
    let script_path = probe_dir.join("run_probe.py");
    crate::write_lf(&script_path, &code)?;

    let before_untracked = git_untracked_files(&env.ws);
    let diff_before = git_diff_head(&env.ws);

    let sp = script_path.display().to_string();
    let cmd = if sp.contains(' ') {
        format!("python3 \"{sp}\"")
    } else {
        format!("python3 {sp}")
    };
    let (rc, out) = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, 15);

    // Clean any stray untracked files the snippet wrote into `/workspace`
    clean_new_untracked(&env.ws, &before_untracked);

    // Capture `/tmp/.agent/repro.py` automatically:
    // - Before patching (`diff_before` empty), if a probe fails with a traceback inside `/workspace`
    // - After patching (`!diff_before.is_empty`), if an assertion probe (`assert ` + `import `) passes (`rc == 0`)
    let repro_path = env.agent_dir.join("repro.py");
    let mut captured_note = String::new();
    if !repro_path.is_file() {
        if rc != 0
            && diff_before.trim().is_empty()
            && !crate::debug::parse_repo_frames(&out, &env.ws).is_empty()
        {
            let _ = crate::write_lf(&repro_path, &code);
            captured_note =
                " (captured as `/tmp/.agent/repro.py` for automatic patch verification)"
                    .to_string();
        } else if rc == 0
            && !diff_before.trim().is_empty()
            && code.contains("import ")
            && code.contains("assert")
        {
            let _ = crate::write_lf(&repro_path, &code);
            captured_note = " (saved passing reproducer to `/tmp/.agent/repro.py`)".to_string();
        }
    }

    let summary = format!(
        "PROBE RESULT (exit {rc}){captured_note}:\n{}",
        tail_lines(&out, 25)
    );
    Ok(clip_tokens(&summary, 450))
}

fn is_transient_adk_script(rel_path: &str) -> bool {
    Path::new(rel_path)
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|n| n.starts_with(".adk_exec"))
}

fn enforce_workspace_hygiene(env: &EnvPaths) -> Vec<String> {
    let mut actions = Vec::new();
    let created_path = env.agent_dir.join("created_files.json");
    let created: BTreeSet<String> = crate::read_lf(&created_path)
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
        .into_iter()
        .collect();

    // 1. Revert any modified test/config files OR destructive mass-deletion files
    for f in git_modified_files(&env.ws) {
        let lower = f.to_ascii_lowercase();
        let is_cfg = matches!(
            Path::new(&lower).file_name().and_then(|s| s.to_str()),
            Some("setup.py" | "setup.cfg" | "pyproject.toml" | "tox.ini" | "conftest.py")
        );
        if is_test_path(&f) || is_cfg {
            let _ = Command::new("git")
                .args(["checkout", "HEAD", "--", &f])
                .current_dir(&env.ws)
                .output();
            actions.push(format!("reverted protected test/config file `{f}`"));
            continue;
        }
        let file_diff = git_diff_file(&env.ws, &f);
        if crate::patch::is_destructive_diff(&file_diff) {
            let _ = Command::new("git")
                .args(["checkout", "HEAD", "--", &f])
                .current_dir(&env.ws)
                .output();
            actions.push(format!(
                "reverted destructive mass-deletion in `{f}` (restoring `best.patch` if available)"
            ));
        }
    }

    // 2. Remove untracked scratch/test files inside `/workspace`, while preserving legitimate
    //    new package modules created via `patch create=1` (staging with `git add -N`).
    for u in git_untracked_files(&env.ws) {
        if is_transient_adk_script(&u) {
            continue;
        }
        if created.contains(&u) || is_legitimate_new_package_module(&u) {
            crate::patch::record_created_file(env, &u);
            continue;
        }
        let p = crate::join_rel(&env.ws, &u);
        if p.is_file() {
            let _ = std::fs::remove_file(&p);
            actions.push(format!("removed untracked file `{u}`"));
        } else if p.is_dir() {
            let _ = std::fs::remove_dir_all(&p);
            actions.push(format!("removed untracked dir `{u}`"));
        }
    }

    actions
}

fn is_legitimate_new_package_module(rel_path: &str) -> bool {
    if !rel_path.ends_with(".py")
        || is_test_path(rel_path)
        || crate::ast_graph::is_docs_path(rel_path)
    {
        return false;
    }
    let segs = crate::path_segments(rel_path);
    if segs.len() < 2
        || segs
            .iter()
            .any(|s| s == "__pycache__" || s.starts_with('.'))
    {
        return false;
    }
    let fname = segs
        .last()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    ![
        "repro", "probe", "scratch", "debug", "tmp_", "temp_", "test_",
    ]
    .iter()
    .any(|pfx| fname.starts_with(pfx))
}

fn run_repro_worker(env: &EnvPaths) -> (Option<bool>, String) {
    let repro_path = env.agent_dir.join("repro.py");
    if !repro_path.is_file() {
        return (None, String::new());
    }
    let probe_dir = env.agent_dir.join("probe");
    let _ = std::fs::create_dir_all(&probe_dir);
    // `from src.pkg import ...` loads a second copy of the package, so the repro would pass or
    // fail independently of the installed one: rewrite the import and add a note.
    let rewrite_note = match crate::read_lf(&repro_path)
        .ok()
        .and_then(|c| rewrite_src_imports(&c, &env.ws))
    {
        Some((fixed, pkg)) => {
            let _ = crate::write_lf(&repro_path, &fixed);
            format!(
                "  NOTE: rewrote `src.{pkg}` to `{pkg}` in your repro (the installed package; `src/` is not a package)"
            )
        }
        None => String::new(),
    };
    let rp = repro_path.display().to_string();
    let cmd = if rp.contains(' ') {
        format!("python3 \"{rp}\"")
    } else {
        format!("python3 {rp}")
    };
    let limit = repro_limit(env, 15);
    let (mut rc, mut out) = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, limit);
    let mut slow_note = String::new();
    if rc == 124
        && let Some(wider) =
            widen_repro_limit(env, limit, &crate::read_lf(&repro_path).unwrap_or_default())
    {
        (rc, out) = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, wider);
        slow_note = format!(
            "  NOTE: importing this package alone takes most of {limit} s here; repros now run with a {wider} s limit"
        );
    }
    if rc == 124
        && let Some(secs) = import_slow_secs(env)
    {
        slow_note = format!(
            "  NOTE: the imports at the top of your repro alone take ~{secs} s in this sandbox, longer than a repro may run, so this TIMEOUT says nothing about your patch. Import only the module you changed (no heavy third-party imports), or rely on the existing tests."
        );
    }
    let limit = repro_limit(env, 15);
    // A third-party import the sandbox lacks stops the repro before it reaches the code under
    // test: re-run with those modules stubbed.
    let issue_low = crate::read_lf(&env.agent_dir.join("issue.txt"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut stubbed: Vec<String> = Vec::new();
    while rc != 0 && stubbed.len() < 4 {
        let Some(m) = external_missing_module(&out, &env.ws) else {
            break;
        };
        if stubbed.contains(&m) || issue_low.contains(&m.to_ascii_lowercase()) {
            break;
        }
        stubbed.push(m);
        let runner = env.agent_dir.join("stubrun.py");
        let _ = crate::write_lf(&runner, &stub_runner_source(&stubbed, &rp));
        let rn = runner.display().to_string();
        (rc, out) =
            run_shell_with_timeout(&format!("python3 \"{rn}\""), &probe_dir, &env.ws, limit);
    }
    let stub_note = if stubbed.is_empty() {
        String::new()
    } else {
        format!(
            "  (ran with stand-ins for third-party module(s) not installed in this sandbox: {}; not your bug, do not patch around it)",
            stubbed
                .iter()
                .map(|m| format!("`{m}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    if rc == 0 {
        let code = crate::read_lf(&repro_path).unwrap_or_default();
        let note = [
            rewrite_note,
            stub_note,
            slow_note,
            vacuous_repro_note(&code),
            private_only_repro_note(&code),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
        (Some(true), note)
    } else {
        let code = crate::read_lf(&repro_path).unwrap_or_default();
        let notes = [
            rewrite_note,
            stub_note,
            slow_note,
            src_import_note(&code, &env.ws),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
        if notes.is_empty() {
            (Some(false), tail_lines(&out, 14))
        } else {
            (
                Some(false),
                format!("{}\n{}", notes.join("\n"), tail_lines(&out, 14)),
            )
        }
    }
}

/// Repro time limit: `base`, raised to the limit measured for this session when the package import alone is slow.
pub fn repro_limit(env: &EnvPaths, base: u64) -> u64 {
    std::fs::read_to_string(env.agent_dir.join("repro_limit_s"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(base, |l| l.max(base))
}

/// The first workspace package the repro imports (`import pkg` / `from pkg.x import y`, at `pkg/` or `src/pkg/`).
fn repro_package(code: &str, ws: &Path) -> Option<String> {
    let re = Regex::new(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)").expect("regex");
    re.captures_iter(code).map(|c| c[1].to_string()).find(|p| {
        ws.join(p).join("__init__.py").is_file()
            || ws.join("src").join(p).join("__init__.py").is_file()
    })
}

/// After a repro of a workspace package times out at `limit`, time its top-level import lines once per
/// session; when they alone take over half the limit, store a longer limit (2 x imports + 10 s, at most 60 s) and
/// return it.
pub fn widen_repro_limit(env: &EnvPaths, limit: u64, code: &str) -> Option<u64> {
    let marker = env.agent_dir.join("repro_limit_s");
    if marker.exists() {
        return None;
    }
    repro_package(code, &env.ws)?;
    // time the repro's own top-level imports, not only the package: a submodule or a third-party
    // import can be the slow one
    let imports: Vec<&str> = code
        .lines()
        .filter(|l| l.starts_with("import ") || l.starts_with("from "))
        .collect();
    let probe_dir = env.agent_dir.join("probe");
    let _ = std::fs::create_dir_all(&probe_dir);
    // a script file, not `python3 -c "..."`: no shell quoting to get wrong
    let script = env.agent_dir.join("import_probe.py");
    crate::write_lf(&script, &(imports.join("\n") + "\n")).ok()?;
    let sp = script.display().to_string();
    let cmd = if sp.contains(' ') {
        format!("python3 \"{sp}\"")
    } else {
        format!("python3 {sp}")
    };
    let t0 = std::time::Instant::now();
    let _ = run_shell_with_timeout(&cmd, &probe_dir, &env.ws, 60);
    let secs = t0.elapsed().as_secs();
    let new = widened_limit(secs, limit);
    let _ = std::fs::write(&marker, new.unwrap_or(limit).to_string());
    if new.is_none() && secs * 2 > limit {
        let _ = std::fs::write(env.agent_dir.join("import_slow_s"), secs.to_string());
    }
    new
}

/// The longer repro limit for imports that take `secs`: 2 x imports + 10 s, at most 60 s, when they eat over half of
/// `limit` and still leave the repro at least 10 s of the 60. Imports that leave no room keep the short limit.
pub fn widened_limit(secs: u64, limit: u64) -> Option<u64> {
    (secs * 2 > limit && secs + 10 <= 60).then_some((secs * 2 + 10).min(60))
}

/// Seconds the repro's imports took when they are too slow for any repro limit (see [`widen_repro_limit`]).
pub fn import_slow_secs(env: &EnvPaths) -> Option<u64> {
    std::fs::read_to_string(env.agent_dir.join("import_slow_s"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// In a `src/` layout the package is installed as `pkg`; `from src.pkg import ...` loads a second
/// copy (or nothing) and the repro fails for reasons unrelated to the patch.
pub fn src_import_note(code: &str, ws: &Path) -> String {
    let re = Regex::new(r"(?m)^\s*(?:from|import)\s+src\.([A-Za-z_][A-Za-z0-9_]*)").expect("regex");
    let Some(pkg) = re.captures(code).map(|c| c[1].to_string()) else {
        return String::new();
    };
    if !ws.join("src").join(&pkg).is_dir() {
        return String::new();
    }
    format!(
        "  NOTE: import the package as `{pkg}` (e.g. `from {pkg}... import ...`), never `src.{pkg}`: the `src/` folder is not a package"
    )
}

/// Root package of the first `ModuleNotFoundError` whose module is not part of the repository.
fn external_missing_module(out: &str, ws: &Path) -> Option<String> {
    let mod_re = Regex::new(r"ModuleNotFoundError: No module named '([^']+)'").expect("regex");
    let root = mod_re
        .captures_iter(out)
        .last()?
        .get(1)?
        .as_str()
        .split('.')
        .next()?
        .to_string();
    let in_repo = root.is_empty()
        || ws.join(&root).exists()
        || ws.join(format!("{root}.py")).exists()
        || ws.join("src").join(&root).exists()
        || ws.join("src").join(format!("{root}.py")).exists();
    (!in_repo).then_some(root)
}

/// Python that installs a first-priority import hook answering `stubbed` packages with permissive
/// stand-ins, then runs `repro` as `__main__`.
pub fn stub_runner_source(stubbed: &[String], repro: &str) -> String {
    let names = stubbed
        .iter()
        .map(|m| format!("{m:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"import importlib.abc, importlib.machinery, runpy, sys, types
STUB = {{{names}}}
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and not k and callable(a[0]) else _Any()
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Any()
    def __iter__(self): return iter(())
    def __bool__(self): return False
    def __mro_entries__(self, bases): return (object,)
def _mod_getattr(n):
    if n.startswith("__"): raise AttributeError(n)
    return _Any()
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = _mod_getattr
        return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Finder())  # whole package, even half-installed (graphtage: scipy)
sys.argv = [{repro:?}]
runpy.run_path({repro:?}, run_name="__main__")
"#
    )
}

/// A repro that exits 0 without asserting anything (no `assert`/`raise`, or an `except` that
/// swallows the failure) proves nothing about the fix.
pub fn vacuous_repro_note(code: &str) -> String {
    let swallows = Regex::new(r"(?m)^\s*except\b[^:\n]*:\s*\n\s*(print|pass)\b")
        .expect("regex")
        .is_match(code);
    if real_check_count(code) > 0 && !swallows {
        return String::new();
    }
    "  WARNING: this repro exits 0 without checking anything (no `assert`, or an `except` that only prints). \
It does NOT show the issue is fixed: rewrite it as straight-line code ending in `assert <expected result>` (no try/except)."
        .to_string()
}

/// `assert`/`raise` lines that check a result: not `assert False`/`raise`, and not inside an `except`
/// block. A repro of the form `try: f(x)` / `except Exception as e: assert False, e` only
/// shows the call does not raise, never that the output is right.
fn real_check_count(code: &str) -> usize {
    let mut except_indent: Option<usize> = None;
    let mut n = 0;
    for line in code.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let indent = line.len() - t.len();
        if except_indent.is_some_and(|ei| indent <= ei) {
            except_indent = None;
        }
        if t.starts_with("except") && t.ends_with(':') {
            except_indent = Some(indent);
            continue;
        }
        if except_indent.is_some() {
            continue;
        }
        let is_check =
            t.starts_with("assert ") || t.starts_with("assert(") || t.starts_with("raise ");
        let trivial = t.starts_with("assert False") || t.starts_with("assert 0") || t == "raise";
        if is_check && !trivial {
            n += 1;
        }
    }
    n
}

/// A repro that imports only private names (`_helper`) checks the function the model just edited, not
/// the behaviour the issue reports through the public API.
pub fn private_only_repro_note(code: &str) -> String {
    let re = Regex::new(r"(?m)^\s*from\s+[A-Za-z_][\w.]*\s+import\s+\(?([^\n)]+)").expect("regex");
    let names: Vec<String> = re
        .captures_iter(code)
        .flat_map(|c| {
            c[1].split(',')
                .map(|n| n.split_whitespace().next().unwrap_or("").to_string())
                .collect::<Vec<_>>()
        })
        .filter(|n| !n.is_empty() && n != "*")
        .collect();
    if names.is_empty() || !names.iter().all(|n| n.starts_with('_')) {
        return String::new();
    }
    "  WARNING: this repro only calls private names, i.e. the code you just edited, not the behaviour the issue reports. \
Call the PUBLIC function/class/CLI the issue uses and assert the result the issue expects."
        .to_string()
}

/// `src.<pkg>` imports in a `src/` layout, rewritten to the installed package `<pkg>`.
pub fn rewrite_src_imports(code: &str, ws: &Path) -> Option<(String, String)> {
    let re = Regex::new(r"\bsrc\.([A-Za-z_][A-Za-z0-9_]*)").expect("regex");
    let pkg = re.captures(code).map(|c| c[1].to_string())?;
    if !ws.join("src").join(&pkg).is_dir() {
        return None;
    }
    let fixed = Regex::new(&format!(r"\bsrc\.{}\b", regex::escape(&pkg)))
        .expect("regex")
        .replace_all(code, pkg.as_str())
        .into_owned();
    (fixed != code).then_some((fixed, pkg))
}

fn select_test_targets(
    ws: &Path,
    index: Option<&crate::ast_graph::RepoIndex>,
    modified_files: &[String],
    explicit_test: &str,
    issue_text: &str,
) -> Vec<String> {
    if !explicit_test.trim().is_empty() {
        return explicit_test
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    // When a repo has no `test_*.py` / `*_test.py` at all, any module under a tests/ directory
    // counts as a test file (pytest runs a file it is given by path).
    let base_of = |c: &str| -> String {
        Path::new(c)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(c)
            .to_string()
    };
    let named = |b: &str| b.starts_with("test_") || b.ends_with("_test.py");
    let named_tests = index.is_none_or(|idx| idx.files.iter().any(|c| named(&base_of(c))));
    let is_test_file = |c: &str| -> bool {
        let b = base_of(c);
        if named_tests {
            named(&b)
        } else {
            is_test_path(c)
                && b.ends_with(".py")
                && b != "__init__.py"
                && b != "conftest.py"
                && !b.starts_with('_')
        }
    };
    let mut candidates = Vec::new();
    if let Some(idx) = index {
        for f in modified_files {
            if let Some(ts) = idx.tests_of.get(f) {
                for t in ts {
                    if !candidates.contains(t) && crate::join_rel(ws, t).is_file() {
                        candidates.push(t.clone());
                    }
                }
            }
            if let Some(defs) = idx.file_defs.get(f) {
                for def_id in defs {
                    if let Some(ts) = idx.tests_of.get(def_id) {
                        for t in ts {
                            if !candidates.contains(t) && crate::join_rel(ws, t).is_file() {
                                candidates.push(t.clone());
                            }
                        }
                    }
                }
            }
        }
        // Fallback 1: stem match across `idx.files` (actual test_*.py / *_test.py files only, excluding docs)
        if candidates.is_empty() {
            for f in modified_files {
                let stem = Path::new(f)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                if stem.is_empty() || stem == "__init__" {
                    continue;
                }
                let t1 = format!("test_{stem}.py");
                let t2 = format!("{stem}_test.py");
                for cand in &idx.files {
                    let base = Path::new(cand)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or(cand);
                    if is_test_path(cand)
                        && is_test_file(cand)
                        && !crate::ast_graph::is_docs_path(cand)
                        && (cand.ends_with(&t1) || cand.ends_with(&t2) || base.contains(stem))
                        && !candidates.contains(cand)
                    {
                        candidates.push(cand.clone());
                    }
                }
            }
        }
        // Always score test files by subword stem + issue token overlap so topic-specific tests
        // (e.g. `test_console.py` or `test_flag_values.py`) rank ahead of arbitrary alphabetical reverse-importers
        let mut query_toks: BTreeSet<String> = BTreeSet::new();
        for f in modified_files {
            for tok in tokenize_ident_and_subwords(f) {
                if tok.len() >= 3 && !matches!(tok.as_str(), "src" | "pkg" | "lib" | "init") {
                    query_toks.insert(tok);
                }
            }
        }
        for tok in tokenize_ident_and_subwords(issue_text) {
            if tok.len() >= 4 {
                query_toks.insert(tok);
            }
        }
        let mut scored_tests: Vec<(usize, String)> = Vec::new();
        for cand in &idx.files {
            let base = Path::new(cand)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(cand);
            if !is_test_path(cand)
                || !is_test_file(cand)
                || crate::ast_graph::is_docs_path(cand)
                || !crate::join_rel(ws, cand).is_file()
            {
                continue;
            }
            let mut overlap = 0usize;
            for tok in tokenize_ident_and_subwords(base) {
                if tok != "test" && tok != "tests" && query_toks.contains(&tok) {
                    overlap += 3;
                }
            }
            if let Some(fdefs) = idx.file_defs.get(cand) {
                for did in fdefs.iter().take(40) {
                    if let Some(dn) = idx.nodes.get(did) {
                        for tok in tokenize_ident_and_subwords(&dn.name) {
                            if tok != "test" && query_toks.contains(&tok) {
                                overlap += 1;
                            }
                        }
                    }
                }
            }
            if overlap > 0 {
                scored_tests.push((overlap, cand.clone()));
            }
        }
        scored_tests.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        if candidates.is_empty() {
            for (_, t) in scored_tests.into_iter().take(3) {
                candidates.push(t);
            }
        } else {
            let mut prioritized: Vec<String> = Vec::new();
            for (score, t) in scored_tests.into_iter().take(2) {
                if score >= 4 && !prioritized.contains(&t) {
                    prioritized.push(t);
                }
            }
            for c in candidates {
                if !prioritized.contains(&c) {
                    prioritized.push(c);
                }
            }
            candidates = prioritized;
        }
    }
    candidates
        .into_iter()
        .filter(|c| is_test_file(c) && !crate::ast_graph::is_docs_path(c))
        .take(4)
        .collect()
}

fn missing_module_is_external(out: &str, ws: &Path) -> bool {
    let mod_re = Regex::new(r"ModuleNotFoundError: No module named '([^']+)'").expect("regex");
    let mut saw_any = false;
    for cap in mod_re.captures_iter(out) {
        saw_any = true;
        let root_pkg = cap[1].split('.').next().unwrap_or("");
        if root_pkg.is_empty() {
            continue;
        }
        if ws.join(root_pkg).exists()
            || ws.join(format!("{root_pkg}.py")).exists()
            || ws.join("src").join(root_pkg).exists()
            || ws.join("src").join(format!("{root_pkg}.py")).exists()
        {
            return false;
        }
    }
    saw_any
}

fn run_pytest_once_with_flags(
    env: &EnvPaths,
    targets: &[String],
    extra_flags: &str,
    timeout_s: u64,
) -> (i32, String) {
    let quoted: Vec<String> = targets
        .iter()
        .map(|t| {
            if t.contains(' ') {
                format!("\"{t}\"")
            } else {
                t.clone()
            }
        })
        .collect();
    let extra = if extra_flags.is_empty() {
        String::new()
    } else {
        format!("{extra_flags} ")
    };
    let cmd = format!(
        "python3 -m pytest {extra}-x -q --tb=short -p no:cacheprovider {}",
        quoted.join(" ")
    );
    run_shell_with_timeout(&cmd, &env.ws, &env.ws, timeout_s)
}

fn run_pytest_once(env: &EnvPaths, targets: &[String], timeout_s: u64) -> (i32, String) {
    run_pytest_once_with_flags(env, targets, "", timeout_s)
}

fn is_env_or_plugin_error(out: &str, ws: &Path) -> bool {
    let missing_fixture = out.contains("fixture '") && out.contains("' not found");
    let runner_or_collect = out.contains("No module named pytest")
        || out.contains("ImportError while loading conftest")
        || (out.contains("ImportError while importing test module")
            && missing_module_is_external(out, ws))
        || (out.contains("ERROR collecting") && missing_module_is_external(out, ws))
        || missing_module_is_external(out, ws);
    (missing_fixture || runner_or_collect)
        && !out.contains("AssertionError")
        && !out.contains("FAILED ")
}

fn extract_surgical_pytest_failure(raw_out: &str) -> String {
    let mut failure_lines = Vec::new();
    let mut failing_test = None;
    let lines: Vec<&str> = raw_out.lines().collect();

    for ln in &lines {
        let t = ln.trim();
        if t.starts_with("FAILED ") {
            failing_test = Some(t.strip_prefix("FAILED ").unwrap_or(t).to_string());
        } else if t.starts_with("____") && t.ends_with("____") {
            failing_test = Some(t.trim_matches('_').trim().to_string());
        }
        if ln.starts_with("E   ") || ln.starts_with("E       ") {
            failure_lines.push(ln.trim_end().to_string());
        }
    }

    if !failure_lines.is_empty() {
        let mut out = String::new();
        if let Some(test_name) = failing_test {
            out.push_str(&format!("FAILING TEST: {test_name}\n"));
        }
        out.push_str("FAILING ASSERTION:\n");
        out.push_str(&failure_lines.join("\n"));
        return out;
    }

    tail_lines(raw_out, 18)
}

/// Seconds one pytest run may take: 40 % of what is left before the session reserve (260 s,
/// scaled), between 15 and 65.
pub fn pytest_budget_secs(elapsed_secs: u64, time_scale: f64) -> u64 {
    let left = ((260.0 * time_scale) as u64).saturating_sub(elapsed_secs);
    (left * 2 / 5).clamp(15, 65)
}

fn run_pytest_worker(
    env: &EnvPaths,
    test_targets: &[String],
    elapsed_secs: u64,
) -> (PytestStatus, String, Vec<String>) {
    if test_targets.is_empty() {
        return (PytestStatus::NoTestsFound, String::new(), Vec::new());
    }
    let timeout_s = pytest_budget_secs(elapsed_secs, crate::budget_scale());
    let (rc, out) = run_pytest_once(env, test_targets, timeout_s);
    if rc == 0 {
        return (
            PytestStatus::Pass,
            tail_lines(&out, 6),
            test_targets.to_vec(),
        );
    }

    if is_env_or_plugin_error(&out, &env.ws) {
        // When `tests/conftest.py` fails to load due to an uninstalled optional external dependency
        // (e.g. `ModuleNotFoundError: No module named 'typer'`), retry with `--noconftest` so standalone
        // unit test files still run
        if out.contains("ImportError while loading conftest")
            && missing_module_is_external(&out, &env.ws)
        {
            let (rc_nc, out_nc) =
                run_pytest_once_with_flags(env, test_targets, "--noconftest", timeout_s.min(25));
            if rc_nc == 0 {
                return (
                    PytestStatus::Pass,
                    tail_lines(&out_nc, 6),
                    test_targets.to_vec(),
                );
            }
            if !is_env_or_plugin_error(&out_nc, &env.ws) {
                // with --noconftest a collection error that exists without the patch is not the patch's FAIL
                return fail_or_preexisting(
                    env,
                    test_targets,
                    &out_nc,
                    "--noconftest",
                    timeout_s.min(25),
                );
            }
        }
        // If multiple test files were selected and one failed due to a missing optional plugin/fixture
        // (e.g. `pytest-mock` `mocker` in `test_commands.py`), retry on the remaining test targets
        if test_targets.len() > 1 {
            let remaining: Vec<String> = test_targets
                .iter()
                .filter(|t| {
                    let fname = Path::new(t.as_str())
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or(t);
                    !out.contains(t.as_str()) && !out.contains(fname)
                })
                .cloned()
                .collect();
            if !remaining.is_empty() && remaining.len() < test_targets.len() {
                let (rc2, out2) = run_pytest_once(env, &remaining, timeout_s.min(25));
                if rc2 == 0 {
                    return (PytestStatus::Pass, tail_lines(&out2, 6), remaining);
                }
                if is_env_or_plugin_error(&out2, &env.ws) {
                    return (PytestStatus::EnvSkip, tail_lines(&out2, 8), remaining);
                }
                return fail_or_preexisting(env, &remaining, &out2, "", timeout_s.min(25));
            }
        }
        return (
            PytestStatus::EnvSkip,
            tail_lines(&out, 8),
            test_targets.to_vec(),
        );
    }

    fail_or_preexisting(env, test_targets, &out, "", timeout_s.min(40))
}

/// `FAILED tests/x.py::t - ...` / `ERROR tests/x.py - ...` ids from pytest's short summary.
pub(crate) fn failing_ids(out: &str) -> BTreeSet<String> {
    out.lines()
        .filter_map(|l| {
            let t = l.trim();
            t.strip_prefix("FAILED ")
                .or_else(|| t.strip_prefix("ERROR "))
                .map(|r| r.split(" - ").next().unwrap_or(r).trim().to_string())
        })
        .filter(|id| !id.is_empty() && !id.starts_with("collecting"))
        .collect()
}

/// A failure that also happens on the unpatched code is not the patch's fault.
/// Runs the same targets once on HEAD
/// (patched files checked out, then the diff re-applied) and caches the ids per target set.
fn fail_or_preexisting(
    env: &EnvPaths,
    targets: &[String],
    out: &str,
    flags: &str,
    timeout_s: u64,
) -> (PytestStatus, String, Vec<String>) {
    let fail = (
        PytestStatus::Fail,
        extract_surgical_pytest_failure(out),
        targets.to_vec(),
    );
    let now = failing_ids(out);
    if now.is_empty() {
        return fail;
    }
    let key = format!("{flags} {}", targets.join(" "));
    let cache = env.agent_dir.join("baseline_pytest.txt");
    let cached = crate::read_lf(&cache).unwrap_or_default();
    let base: BTreeSet<String> = if let Some(line) = cached
        .lines()
        .find(|l| l.split('\t').next() == Some(key.as_str()))
    {
        line.split('\t').skip(1).map(str::to_string).collect()
    } else {
        // at the budget floor there is no time to re-run the tests on the unpatched tree
        if timeout_s <= 15 {
            return fail;
        }
        let Some(ids) = baseline_failing_ids(env, targets, flags, timeout_s) else {
            return fail;
        };
        let row = std::iter::once(key.clone())
            .chain(ids.iter().cloned())
            .collect::<Vec<_>>()
            .join("\t");
        let _ = crate::write_lf(&cache, &format!("{cached}{row}\n"));
        ids
    };
    if now.is_subset(&base) {
        return (
            PytestStatus::EnvSkip,
            format!(
                "PRE-EXISTING: {} also fail(s) WITHOUT your patch (not caused by it; ignore them)",
                now.into_iter().take(3).collect::<Vec<_>>().join(", ")
            ),
            targets.to_vec(),
        );
    }
    fail
}

fn baseline_failing_ids(
    env: &EnvPaths,
    targets: &[String],
    flags: &str,
    timeout_s: u64,
) -> Option<BTreeSet<String>> {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(&env.ws)
            .output()
            .ok()
    };
    let diff = git(&["-c", "core.fileMode=false", "diff", "--binary", "HEAD"])?;
    if !diff.status.success() || diff.stdout.is_empty() {
        return None;
    }
    let saved = env.agent_dir.join("baseline_stash.patch");
    std::fs::write(&saved, &diff.stdout).ok()?;
    let names = git(&["diff", "--name-only", "HEAD"])?;
    let files: Vec<String> = String::from_utf8_lossy(&names.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    let mut co = vec!["checkout", "HEAD", "--"];
    co.extend(files.iter().map(String::as_str));
    git(&co)?;
    let (_, base_out) = run_pytest_once_with_flags(env, targets, flags, timeout_s);
    restore_baseline_stash(env);
    Some(failing_ids(&base_out))
}

/// Put the patch set aside by [`baseline_failing_ids`] back and drop the saved copy. Called at the start of every
/// call too: a verify killed while the baseline tests ran on the unpatched files would otherwise
/// leave the workspace without the edit. Returns whether it restored.
pub fn restore_baseline_stash(env: &EnvPaths) -> bool {
    let saved = env.agent_dir.join("baseline_stash.patch");
    if !saved.is_file() {
        return false;
    }
    let apply = |p: &Path| {
        Command::new("git")
            .args(["apply", "--whitespace=nowarn", &p.display().to_string()])
            .current_dir(&env.ws)
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if !apply(&saved) {
        // never leave the workspace unpatched: fall back to the last clean snapshot
        let _ = apply(&env.agent_dir.join("best.patch"));
    }
    let _ = std::fs::remove_file(&saved);
    true
}

fn check_issue_literals(env: &EnvPaths, diff_text: &str) -> Vec<String> {
    let Ok(issue) = crate::read_lf(&env.agent_dir.join("issue.txt")) else {
        return Vec::new();
    };
    let mut notes = Vec::new();
    let patterns = [
        r#"(?:ValueError|TypeError|KeyError|RuntimeError|AttributeError|ValidationError|AssertionError)\(\s*['"]([^'"]{6,80})['"]\s*\)"#,
        r#"match\s*=\s*['"]([^'"]{6,80})['"]"#,
        r#"assert\s+[^,\n]+,\s*['"]([^'"]{6,80})['"]"#,
    ];
    let diff_low = diff_text.to_ascii_lowercase();
    for pat in patterns {
        if let Ok(re) = Regex::new(pat) {
            for cap in re.captures_iter(&issue) {
                let phrase = &cap[1];
                let phrase_low = phrase.to_ascii_lowercase();
                if !diff_low.contains(&phrase_low)
                    && !notes.iter().any(|n: &String| n.contains(phrase))
                {
                    notes.push(format!(
                        "issue/test expects exact message: `\"{phrase}\"` — verify if your patch should match this exact string"
                    ));
                    if notes.len() >= 2 {
                        return notes;
                    }
                }
            }
        }
    }
    notes
}

fn git_diff_head(ws: &Path) -> String {
    Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "HEAD"])
        .current_dir(ws)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

fn git_modified_files(ws: &Path) -> Vec<String> {
    Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "--name-only", "HEAD"])
        .current_dir(ws)
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Root-level scratch files the agent made with raw shell (`cat > repro.py`, `> out.txt`) are moved to
/// `<tmp>/scratch/` so they never enter the diff.
/// Only untracked files directly at the root with a scratch-like name; package files are never touched.
pub fn move_root_scratch(ws: &Path, tmp: &Path) -> Vec<String> {
    let re = regex::Regex::new(
        r"(?i)^(repro|reproduce|reproduction|scratch|debug|check|try|tmp|temp|out|output|test_repro|test_issue|issue)[\w.-]*\.(py|txt|json|log|out|md|yaml|yml|csv|html|xml)$|^(a|b|c|x|y|file\d*|input\d*|output\d*)\.(txt|json|csv|log|yaml|yml)$|\.log$",
    )
    .expect("regex");
    let mut moved = Vec::new();
    for f in git_untracked_files(ws) {
        if f.contains('/') || !re.is_match(&f) {
            continue;
        }
        let dst_dir = tmp.join("scratch");
        if std::fs::create_dir_all(&dst_dir).is_ok()
            && std::fs::rename(ws.join(&f), dst_dir.join(&f)).is_ok()
        {
            moved.push(f);
        }
    }
    moved
}

fn git_untracked_files(ws: &Path) -> BTreeSet<String> {
    Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(ws)
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn git_diff_file(ws: &Path, rel_file: &str) -> String {
    Command::new("git")
        .args(["-c", "core.fileMode=false", "diff", "HEAD", "--", rel_file])
        .current_dir(ws)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

fn clean_new_untracked(ws: &Path, before: &BTreeSet<String>) {
    for u in git_untracked_files(ws) {
        if is_transient_adk_script(&u) {
            continue;
        }
        if !before.contains(&u) {
            let p = crate::join_rel(ws, &u);
            if p.is_file() {
                let _ = std::fs::remove_file(p);
            } else if p.is_dir() {
                let _ = std::fs::remove_dir_all(p);
            }
        }
    }
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a repo whose tests carry no `test_` prefix still gets its tests run by verify.
    #[test]
    fn unprefixed_test_modules_are_selected_when_the_repo_has_no_named_tests() {
        let root = tempfile::TempDir::new().expect("tmp");
        let ws = root.path().join("workspace");
        let tmp = root.path().join("tmp");
        std::fs::create_dir_all(ws.join("vermin")).unwrap();
        std::fs::create_dir_all(ws.join("tests")).unwrap();
        std::fs::create_dir_all(&tmp).unwrap();
        crate::write_lf(&ws.join("vermin").join("__init__.py"), "").unwrap();
        crate::write_lf(
            &ws.join("vermin").join("source_visitor.py"),
            "def visit_function(node):\n    return node\n",
        )
        .unwrap();
        crate::write_lf(&ws.join("tests").join("__init__.py"), "").unwrap();
        crate::write_lf(
            &ws.join("tests").join("function.py"),
            "from vermin.source_visitor import visit_function\n\ndef test_function():\n    assert visit_function(1) == 1\n",
        )
        .unwrap();
        let env = EnvPaths::with_roots(ws.clone(), tmp);
        let idx = crate::load_or_build_index(&env).expect("index");
        let got = select_test_targets(
            &ws,
            Some(&idx),
            &["vermin/source_visitor.py".to_string()],
            "",
            "visit_function returns the wrong function node",
        );
        assert!(
            got.iter().any(|t| t.ends_with("tests/function.py")),
            "{got:?}"
        );
        assert!(!got.iter().any(|t| t.ends_with("__init__.py")), "{got:?}");
    }
}
