//! `setup`'s HTTP downloads, which can't hang: connecting has a time limit ([`CONNECT_TIMEOUT`]), a
//! download that receives nothing for [`STALL_TIMEOUT`] fails (one that's slow but still receiving
//! never does: there's no limit on the whole body), and a transient failure is retried after
//! [`RETRY_DELAYS`], resuming where it stopped with an HTTP Range request (from the start if the
//! server doesn't do ranges). The caller checks the whole file's SHA-256 as before. The same values
//! as aji-runtime's downloads (its `fetch.rs`).

use std::cell::Cell;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;
use std::time::Duration;

const UA: &str = concat!("mangajanai/", env!("CARGO_PKG_VERSION"));

/// Resolving the host and connecting (with the TLS handshake), each.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// A download (or a response's headers) that receives no bytes for this long fails, and is retried.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// The waits before each retry after a failure that received nothing new: 4 retries in about a
/// minute. A failure after new bytes starts this over (a long download on a flaky connection keeps
/// going while it gets somewhere).
pub const RETRY_DELAYS: [Duration; 4] =
    [Duration::from_secs(2), Duration::from_secs(5), Duration::from_secs(15), Duration::from_secs(30)];
/// Attempts per download in all, however much each one received.
pub const MAX_ATTEMPTS: u32 = 20;

/// The timeouts and retries of a download (tests use short ones).
#[derive(Clone, Debug)]
pub struct Policy {
    pub connect: Duration,
    pub stall: Duration,
    pub retry_delays: Vec<Duration>,
    pub max_attempts: u32,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy { connect: CONNECT_TIMEOUT, stall: STALL_TIMEOUT, retry_delays: RETRY_DELAYS.to_vec(), max_attempts: MAX_ATTEMPTS }
    }
}

fn agent(policy: &Policy) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_resolve(Some(policy.connect))
        .timeout_connect(Some(policy.connect))
        // restarts with every read: a stall, not a slow download
        .timeout_per_read(Some(policy.stall))
        .timeout_per_write(Some(policy.stall))
        .build()
        .into()
}

fn secs(d: Duration) -> String {
    if d >= Duration::from_secs(1) { format!("{} s", d.as_secs()) } else { format!("{} ms", d.as_millis()) }
}

/// Why an attempt failed, and whether trying again may help.
enum Failure {
    Retry(String),
    Fatal(String),
}

/// A failed request or body read: timeouts, dropped connections and server errors are worth another
/// try; a missing file, a refused request or an unknown host (offline) aren't.
fn failure(e: ureq::Error, policy: &Policy) -> Failure {
    use ureq::Error as E;
    match e {
        E::Timeout(ureq::Timeout::Resolve | ureq::Timeout::Connect) => {
            Failure::Retry(format!("couldn't connect within {}", secs(policy.connect)))
        }
        E::Timeout(_) => Failure::Retry(format!("no data for {}", secs(policy.stall))),
        E::StatusCode(c) if c == 408 || c == 429 || c >= 500 => Failure::Retry(format!("the server answered {c}")),
        E::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof => Failure::Retry("the connection closed early".into()),
        e @ (E::Io(_) | E::ConnectionFailed | E::Protocol(_)) => Failure::Retry(e.to_string()),
        e => Failure::Fatal(e.to_string()),
    }
}

/// Runs `attempt` until it succeeds, a failure isn't transient, or the retries run out. `progress`
/// is how far the download is (more than before starts the retry delays over).
fn with_retries<T>(
    url: &str,
    policy: &Policy,
    mut attempt: impl FnMut(&ureq::Agent) -> Result<T, Failure>,
    progress: impl Fn() -> u64,
) -> Result<T, String> {
    let agent = agent(policy);
    let name = url.rsplit('/').next().unwrap_or(url);
    let (mut attempts, mut fails) = (0u32, 0usize);
    loop {
        let before = progress();
        attempts += 1;
        let why = match attempt(&agent) {
            Ok(v) => return Ok(v),
            Err(Failure::Fatal(why)) => return Err(format!("downloading {url}: {why}")),
            Err(Failure::Retry(why)) => why,
        };
        let now = progress();
        if now > before {
            fails = 0;
        }
        if fails >= policy.retry_delays.len() || attempts >= policy.max_attempts {
            return Err(format!("downloading {url}: {why} (gave up after {attempts} attempts)"));
        }
        let wait = policy.retry_delays[fails];
        fails += 1;
        let from = if now > 0 { format!(", resuming at {:.0} MB", now as f64 / 1e6) } else { String::new() };
        if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            eprintln!(); // ends a progress line
        }
        eprintln!("  {name}: {why}; trying again in {}{from}", secs(wait));
        std::thread::sleep(wait);
    }
}

/// Fetches a small file (the runtime index) into memory.
pub fn fetch_small(url: &str, policy: &Policy) -> Result<Vec<u8>, String> {
    with_retries(
        url,
        policy,
        |agent| {
            let mut resp = agent.get(url).header("User-Agent", UA).call().map_err(|e| failure(e, policy))?;
            resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| failure(e, policy))
        },
        || 0,
    )
}

/// Downloads `url` to `dest`, calling `progress(got, total)` as bytes arrive (the caller limits how
/// often it shows them). `expected` is the size the pin says: a download that ends short of it is
/// resumed.
pub fn download(url: &str, dest: &Path, expected: u64, policy: &Policy, progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let file = std::fs::File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    // bytes in `dest` so far: shared by the attempts and the retry loop
    let got = Cell::new(0u64);
    with_retries(url, policy, |agent| attempt(agent, url, dest, expected, policy, &mut out, &got, progress), || got.get())?;
    out.flush().map_err(|e| format!("{}: {e}", dest.display()))
}

/// One request: from the start, or from `got` with a Range request. Appends to `out`.
#[allow(clippy::too_many_arguments)]
fn attempt(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    expected: u64,
    policy: &Policy,
    out: &mut BufWriter<std::fs::File>,
    got: &Cell<u64>,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), Failure> {
    let disk = |e: std::io::Error| Failure::Fatal(format!("{}: {e}", dest.display()));
    let from = got.get();
    // Range offsets are of the bytes as sent: no compression
    let mut req = agent.get(url).header("User-Agent", UA).header("Accept-Encoding", "identity");
    if from > 0 {
        req = req.header("Range", format!("bytes={from}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // what's there is already the whole file (resumed exactly at its end)
        Err(ureq::Error::StatusCode(416)) if from > 0 && from == expected => return Ok(()),
        Err(ureq::Error::StatusCode(416)) => {
            restart(out, got).map_err(disk)?;
            return Err(Failure::Retry("the server refused to resume; starting over".into()));
        }
        Err(e) => return Err(failure(e, policy)),
    };
    let header = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let length = header("content-length").and_then(|v| v.parse::<u64>().ok());
    let resumed = from > 0 && resp.status().as_u16() == 206;
    if resumed && !header("content-range").is_some_and(|r| r.starts_with(&format!("bytes {from}-"))) {
        restart(out, got).map_err(disk)?;
        return Err(Failure::Retry("the server resumed at the wrong place; starting over".into()));
    }
    if from > 0 && !resumed {
        // the server sent the whole file: start over
        restart(out, got).map_err(disk)?;
    }
    let total = match length {
        Some(n) => got.get() + n,
        None => expected,
    };
    progress(got.get(), total);
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(e) => return Err(failure(ureq::Error::from(e), policy)),
        };
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(disk)?;
        got.set(got.get() + n as u64);
        progress(got.get(), total);
    }
    // without a Content-Length, a dropped connection looks like the end: the pin says how big
    if got.get() < expected {
        return Err(Failure::Retry(format!("the connection closed early ({} of {expected} bytes)", got.get())));
    }
    Ok(())
}

/// Empties the download to start it over.
fn restart(out: &mut BufWriter<std::fs::File>, got: &Cell<u64>) -> std::io::Result<()> {
    out.flush()?;
    out.get_mut().set_len(0)?;
    out.seek(std::io::SeekFrom::Start(0))?;
    got.set(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Against a local server that misbehaves on purpose.
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug)]
    enum Serve {
        /// The whole body (or from the Range asked for).
        Whole,
        /// Headers with the full Content-Length, then this many bytes, then the connection is closed.
        DropAfter(usize),
        /// Headers, this many bytes, then nothing (the connection stays open).
        StallAfter(usize),
        /// The whole body with 200, whatever Range was asked for.
        IgnoreRange,
        Status(u16),
    }

    struct TestServer {
        url: String,
        /// The Range header of each request, in order.
        ranges: Arc<Mutex<Vec<Option<String>>>>,
    }

    /// Serves `body`; the n-th connection is answered as `script[n]` (the last entry for the rest).
    fn serve(body: Vec<u8>, script: Vec<Serve>) -> TestServer {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/pack.7z", l.local_addr().unwrap());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let seen = ranges.clone();
        let body = Arc::new(body);
        std::thread::spawn(move || {
            for (i, conn) in l.incoming().enumerate() {
                let Ok(mut s) = conn else { continue };
                let how = *script.get(i).or(script.last()).unwrap();
                let (body, seen) = (body.clone(), seen.clone());
                std::thread::spawn(move || {
                    let mut req = Vec::new();
                    let mut b = [0u8; 1];
                    while !req.ends_with(b"\r\n\r\n") && s.read(&mut b).unwrap_or(0) == 1 {
                        req.push(b[0]);
                    }
                    let req = String::from_utf8_lossy(&req).to_ascii_lowercase();
                    let range = req.lines().find_map(|l| l.strip_prefix("range: ")).map(|r| r.trim().to_string());
                    seen.lock().unwrap().push(range.clone());
                    let from = range
                        .as_deref()
                        .and_then(|r| r.strip_prefix("bytes="))
                        .and_then(|r| r.strip_suffix('-'))
                        .and_then(|n| n.parse::<usize>().ok())
                        .filter(|_| !matches!(how, Serve::IgnoreRange))
                        .unwrap_or(0);
                    if let Serve::Status(c) = how {
                        let _ = write!(s, "HTTP/1.1 {c} Nope\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        return;
                    }
                    let part = &body[from..];
                    let head = if from > 0 {
                        format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\n", body.len() - 1, body.len())
                    } else {
                        "HTTP/1.1 200 OK\r\n".to_string()
                    };
                    let _ = write!(s, "{head}Content-Length: {}\r\nConnection: close\r\n\r\n", part.len());
                    match how {
                        Serve::DropAfter(n) => {
                            let _ = s.write_all(&part[..n.min(part.len())]);
                        }
                        Serve::StallAfter(n) => {
                            let _ = s.write_all(&part[..n.min(part.len())]);
                            let _ = s.flush();
                            std::thread::sleep(Duration::from_secs(5));
                        }
                        _ => {
                            let _ = s.write_all(part);
                        }
                    }
                });
            }
        });
        TestServer { url, ranges }
    }

    fn quick() -> Policy {
        Policy { connect: Duration::from_secs(5), stall: Duration::from_millis(300), retry_delays: vec![Duration::from_millis(20); 4], max_attempts: 20 }
    }

    fn body() -> Vec<u8> {
        (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    fn fetch(name: &str, srv: &TestServer, expected: u64) -> (Result<(), String>, Vec<u8>) {
        let dir = std::env::temp_dir().join(format!("mangajanai-download-test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("pack.7z.part");
        let r = download(&srv.url, &dest, expected, &quick(), &mut |_, _| {});
        let got = std::fs::read(&dest).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        (r, got)
    }

    #[test]
    fn a_dropped_download_resumes_where_it_stopped() {
        let b = body();
        let srv = serve(b.clone(), vec![Serve::DropAfter(70_000), Serve::Whole]);
        let (r, got) = fetch("drop", &srv, b.len() as u64);
        r.unwrap();
        assert!(got == b);
        assert_eq!(*srv.ranges.lock().unwrap(), vec![None, Some("bytes=70000-".to_string())]);
    }

    #[test]
    fn a_stalled_download_times_out_and_resumes() {
        let b = body();
        let srv = serve(b.clone(), vec![Serve::StallAfter(50_000), Serve::Whole]);
        let start = std::time::Instant::now();
        let (r, got) = fetch("stall", &srv, b.len() as u64);
        r.unwrap();
        assert!(got == b);
        assert_eq!(*srv.ranges.lock().unwrap(), vec![None, Some("bytes=50000-".to_string())]);
        assert!(start.elapsed() < Duration::from_secs(4), "it gave up on the stall, not the server");
    }

    #[test]
    fn a_slow_download_that_keeps_coming_doesnt_time_out() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/slow.7z", l.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut b = [0u8; 4096];
            let _ = s.read(&mut b);
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: 10000\r\nConnection: close\r\n\r\n");
            for _ in 0..10 {
                std::thread::sleep(Duration::from_millis(200));
                let _ = s.write_all(&[7u8; 1000]);
                let _ = s.flush();
            }
        });
        let dir = std::env::temp_dir().join(format!("mangajanai-download-test-slow-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("slow.part");
        download(&url, &dest, 10_000, &Policy { retry_delays: Vec::new(), ..quick() }, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), vec![7u8; 10_000]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_server_without_ranges_starts_over() {
        let b = body();
        let srv = serve(b.clone(), vec![Serve::DropAfter(70_000), Serve::IgnoreRange]);
        let (r, got) = fetch("norange", &srv, b.len() as u64);
        r.unwrap();
        assert!(got == b, "not the first part twice");
    }

    #[test]
    fn a_download_that_never_comes_gives_up() {
        let srv = serve(body(), vec![Serve::StallAfter(0)]);
        let (r, _) = fetch("dead", &srv, 200_000);
        let e = r.err().unwrap();
        assert!(e.contains("no data for") && e.contains("gave up after 5 attempts"), "{e}");
        assert_eq!(srv.ranges.lock().unwrap().len(), 5);
    }

    #[test]
    fn a_download_making_progress_keeps_retrying() {
        let b = body();
        let srv = serve(b.clone(), vec![Serve::DropAfter(30_000)]);
        let (r, got) = fetch("flaky", &srv, b.len() as u64);
        r.unwrap();
        assert!(got == b);
        assert_eq!(srv.ranges.lock().unwrap().len(), 7);
    }

    #[test]
    fn missing_files_and_server_errors() {
        let srv = serve(body(), vec![Serve::Status(404)]);
        let e = fetch("404", &srv, 200_000).0.err().unwrap();
        assert!(e.contains("404"), "{e}");
        assert_eq!(srv.ranges.lock().unwrap().len(), 1, "a 404 isn't retried");
        let b = body();
        let srv = serve(b.clone(), vec![Serve::Status(503), Serve::Whole]);
        assert!(fetch("503", &srv, b.len() as u64).1 == b);
        let srv = serve(b"{\"packs\": []}".to_vec(), vec![Serve::Status(502), Serve::DropAfter(3), Serve::Whole]);
        assert_eq!(fetch_small(&srv.url, &quick()).unwrap(), b"{\"packs\": []}");
        assert_eq!(srv.ranges.lock().unwrap().len(), 3);
    }
}
