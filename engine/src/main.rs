use std::io::{IsTerminal, Read};
use std::sync::mpsc;
use std::time::Duration;

fn read_stdin_nonblocking() -> String {
    if std::io::stdin().is_terminal() {
        return String::new();
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut handle = std::io::stdin().lock();
        let mut chunk = [0u8; 8192];
        loop {
            match handle.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if tx.send(Some(buf.clone())).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(None);
    });

    let mut latest = Vec::new();
    // Wait up to 40ms for the first chunk (heredoc / closed pipe is ready in <1ms;
    // an inherited open parent pipe with no writer times out in 40ms instead of hanging).
    if let Ok(Some(b)) = rx.recv_timeout(Duration::from_millis(40)) {
        latest = b;
        while let Ok(Some(b2)) = rx.recv_timeout(Duration::from_millis(200)) {
            latest = b2;
        }
    }
    String::from_utf8_lossy(&latest).replace("\r\n", "\n")
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let stdin_str = read_stdin_nonblocking();

    match swe_coder::run_cli(&argv, &stdin_str) {
        Ok(out) => {
            let norm = out.replace("\r\n", "\n");
            if norm.ends_with('\n') {
                print!("{norm}");
            } else {
                println!("{norm}");
            }
        }
        Err(e) => {
            let norm = format!("ERROR: {e}").replace("\r\n", "\n");
            eprintln!("{norm}");
            std::process::exit(1);
        }
    }
}
