import numpy as np
#----------------------------------------- funções scraping ---------------------------------------------------------------

from selenium import webdriver
import time, random, re, shutil, zipfile, subprocess, tempfile, io, sys, math
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path


import requests
from urllib.parse import urlparse, unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from typing import Literal, Optional
from unidecode import unidecode


from urllib.parse import urljoin



def contrato_links_base(contract_id: int,
                        driver,
                        timeout: int = 10,
                        max_tries: int = 3,
                        backoff_start: float = 1.2,
                        fetch_anuncio_pdf: bool = True):
    """
    Carrega a página do contrato UMA vez e extrai:
      - link das 'Peças do procedimento' (se existir)
      - link final PDF do 'Anúncio' (Se fetch_anuncio_pdf=True):
          vai para a página de anúncio e lê 'Ligação para o anúncio'
    Retorna:
      {
        "ok": bool,
        "contract_id": int,
        "pecas_link": str|None,
        "anuncio_pdf": str|None,
        "via": "selenium"
      }
    """
    base_url = "https://www.base.gov.pt"
    contrato_url = f"{base_url}/Base4/pt/detalhe/?type=contratos&id={contract_id}"
    wait = WebDriverWait(driver, timeout)
    backoff = backoff_start

    pecas_link = None
    anuncio_pdf = None

    for attempt in range(1, max_tries + 1):
        try:
            # 1) CONTRATO PAGE
            driver.get(contrato_url)
            wait.until(EC.presence_of_element_located((By.ID, "no-more-tables-mx767")))

            # Peças do procedimento
            try:
                a_pecas = wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'td[data-title="Peças do procedimento"] a')
                ))
                href = (a_pecas.get_attribute("href") or "").strip()
                if href:
                    pecas_link = href if href.lower().startswith("http") else urljoin(base_url, href)
            except Exception:
                # célula pode estar vazia
                pass

            # 2) ANÚNCIO DETAIL → PDF
            if fetch_anuncio_pdf:
                anuncio_url = None
                try:
                    a_anuncio = wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'td[data-title="Anúncio"] a')
                    ))
                    raw = (a_anuncio.get_attribute("href") or "").strip()
                    if raw:
                        anuncio_url = raw if raw.lower().startswith("http") else urljoin(base_url, raw)
                except Exception:
                    anuncio_url = None

                if anuncio_url:
                    driver.get(anuncio_url)

                    # tentar botão/link "Ligação para o anúncio"
                    try:
                        el = wait.until(EC.presence_of_element_located((
                            By.XPATH,
                            "//*[(self::a or self::button) and "
                            "contains(translate(normalize-space(.),"
                            " 'ÂÃÁÀÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç','AAAAEEIOOOUUCAAAEEIOOOUUC'),"
                            " 'ligacao para o anuncio')]"
                        )))
                        href = (el.get_attribute("href") or "").strip()
                        if href and not href.lower().startswith("http"):
                            href = urljoin(anuncio_url, href)
                        if href:
                            anuncio_pdf = href
                    except Exception:
                        # fallback: qualquer <a> que termine em .pdf ou contenha 'dre.pt'
                        for el in driver.find_elements(By.CSS_SELECTOR, "a[href]"):
                            h = (el.get_attribute("href") or "").strip()
                            if not h:
                                continue
                            low = h.lower()
                            if low.endswith(".pdf") or "dre.pt" in low:
                                anuncio_pdf = h
                                break

            # success path (não precisamos que ambos existam)
            return {
                "ok": bool(pecas_link or anuncio_pdf),
                "contract_id": contract_id,
                "pecas_link": pecas_link,
                "anuncio_pdf": anuncio_pdf,
                "via": "selenium",
            }

        except Exception:
            # pequena espera e nova tentativa
            time.sleep(backoff + random.uniform(0.2, 0.8))
            backoff *= 1.7

    # esgotou tentativas
    return {
        "ok": False,
        "contract_id": contract_id,
        "pecas_link": pecas_link,
        "anuncio_pdf": anuncio_pdf,
        "via": "selenium",
    }


def make_js_driver(headless: bool = True, window="1280,900", page_timeout=15):
    """Chrome “magro” para JS: bloqueia imagens/css/fontes, eager load, timeouts curtos."""
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={window}")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    # Estratégia mais rápida (não espera sub-recursos todos)
    opts.page_load_strategy = "eager"

    # Desativar imagens (grande ganho)
    prefs = {"profile.managed_default_content_settings.images": 2}
    opts.add_experimental_option("prefs", prefs)

    drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    drv.set_page_load_timeout(page_timeout)

    # Bloquear recursos pesados via CDP (se disponível)
    try:
        drv.execute_cdp_cmd("Network.enable", {})
        drv.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
                     "*.css", "*.woff", "*.woff2", "*.ttf", "*.otf"]
        })
    except Exception:
        pass
    return drv



BATCH_SIZE = 50          # try 80; if sluggish, drop to 50
SLEEP_BETWEEN = (0.15, 0.45)
CLEAR_EVERY = 30         # clear Chrome cache every N items
MAX_CONSEC_FAIL = 4      # recycle driver if too many in a row

def fetch_contract_links_in_batches(contract_ids):
    results = []
    for start in range(0, len(contract_ids), BATCH_SIZE):
        batch = contract_ids[start:start + BATCH_SIZE]
        drv = make_js_driver(headless=True)  # lean driver
        try:
            consec_fail = 0
            for i, cid in enumerate(batch, 1):
                try:
                    res = contrato_links_base(
                        cid, driver=drv, timeout=10, max_tries=3, fetch_anuncio_pdf=True
                    )
                    results.append(res)
                    print(res)
                    consec_fail = 0
                except Exception as e:
                    results.append({
                        "ok": False, "contract_id": cid,
                        "pecas_link": None, "anuncio_pdf": None,
                        "via": "selenium", "error": repr(e)
                    })
                    consec_fail += 1
                    if consec_fail >= MAX_CONSEC_FAIL:
                        # unhealthy session → break batch and start a fresh driver
                        break

                if i % CLEAR_EVERY == 0:
                    try:
                        drv.execute_cdp_cmd("Network.clearBrowserCache", {})
                        drv.execute_cdp_cmd("Network.clearBrowserCookies", {})
                    except Exception:
                        pass

                time.sleep(random.uniform(*SLEEP_BETWEEN))

        finally:
            drv.quit()

        # if we broke early due to repeated failures, continue with a fresh driver
        continue
    return results



# caso download direto das peças do procedimento


def _safe_name(s: str) -> str:
    s = (s or "").strip() or "file"
    return re.sub(r'[\\/*?:"<>|]+', "_", s)

def _unique_path(base_dir: Path, name: str) -> Path:
    p = base_dir / name
    if not p.exists():
        return p
    stem, ext = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        q = base_dir / f"{stem} ({i}){ext}"
        if not q.exists():
            return q
        i += 1

def _make_session():
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Referer": "https://www.acingov.pt/",
        "Accept": "*/*",
    })
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "HEAD", "OPTIONS"},
        respect_retry_after_header=True,
        raise_on_redirect=True
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def _pick_filename(resp, url_tail: str) -> str:
    # Prefer Content-Disposition (supports RFC5987 filename*=UTF-8'')
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        fname = unquote(m.group(1))
    else:
        fname = Path(url_tail).name or "pecas"
    # If server didn’t give an extension, guess from Content-Type
    if "." not in fname:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "zip" in ctype:
            ext = ".zip"
        elif "pdf" in ctype:
            ext = ".pdf"
        elif "jnlp" in ctype:
            ext = ".jnlp"
        else:
            ext = ".bin"
        fname += ext
    return _safe_name(fname)

def _download_with_resume(sess, url, out_path: Path,
                          chunk_size=64 * 1024,
                          connect_timeout=5,
                          read_timeout=180,
                          max_attempts=8,
                          max_total_time=None):
    """
    Stream download with resume (HTTP Range).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    pos = tmp_path.stat().st_size if tmp_path.exists() else 0
    start_time = time.time()
    attempts = 0
    etag = None
    headers = {}

    while True:
        if max_total_time and (time.time() - start_time) > max_total_time:
            raise TimeoutError(f"Aborted after {max_total_time}s total time: {url}")

        range_hdr = {"Range": f"bytes={pos}-"} if pos > 0 else {}
        resp = sess.get(url, stream=True, headers=range_hdr | headers,
                        timeout=(connect_timeout, read_timeout))

        # If resuming but server ignored Range, restart from scratch
        if pos > 0 and resp.status_code == 200:
            pos = 0
            tmp_path.write_bytes(b"")

        # Capture ETag to protect resumes
        if etag is None and resp.headers.get("ETag"):
            etag = resp.headers["ETag"]
            headers["If-Range"] = etag

        try:
            with tmp_path.open("ab") as f:
                for chunk in resp.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
                        pos += len(chunk)
            break  # success
        except (requests.ReadTimeout, requests.ConnectionError, requests.ChunkedEncodingError) as e:
            attempts += 1
            resp.close()
            if attempts > max_attempts:
                raise ConnectionError(f"Exceeded retries while downloading {url}") from e
            time.sleep(min(0.25 * (2 ** min(attempts, 6)), 10))  # backoff
            continue
        finally:
            resp.close()

    tmp_path.replace(out_path)
    return out_path

def _extract_top_level_zip(zip_path: Path, out_dir: Path) -> list[str]:
    """
    Extract only THIS zip (no nested zip handling). Flatten paths, safe names.
    Delete the zip after successful extraction. Return list of saved file paths.
    """
    saved = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = _safe_name(Path(info.filename).name)  # flatten + sanitize
            dest = _unique_path(out_dir, name)          # avoid clashes
            with zf.open(info, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            saved.append(str(dest))
    # delete archive after success
    try:
        zip_path.unlink()
    except Exception:
        pass
    return saved

# ---------- main ----------

def download_pecas_flat(row, out_root="../docs", timeout=180):
    """
    Downloads row['link'] into ../docs/<id>/ and, if it's a ZIP (by content),
    extracts ONLY that top-level zip into the same folder. Other archive types
    (.rar/.7z) are left as files for a later post-pass you already have.
    """
    out_dir = Path(out_root) / str(row['N.º Procedimento (ID BASE)'])
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = _make_session()
    url = row['URL Peças do Procedimento']
    connect_t, read_t = (5, timeout) if not isinstance(timeout, (tuple, list)) else tuple(timeout)

    # Try to determine a filename early
    try:
        h = sess.head(url, allow_redirects=True, timeout=(connect_t, min(read_t, 30)))
        h.raise_for_status()
        fname = _pick_filename(h, urlparse(url).path)
    except Exception:
        fname = None

    if not fname:
        g = sess.get(url, stream=True, timeout=(connect_t, min(read_t, 50)))
        g.raise_for_status()
        fname = _pick_filename(g, urlparse(url).path)
        g.close()

    out_path = _unique_path(out_dir, _safe_name(fname))

    # Download (with resume)
    _download_with_resume(
        sess, url, out_path,
        chunk_size=64 * 1024,
        connect_timeout=connect_t,
        read_timeout=read_t,
        max_attempts=8,
        max_total_time=None,
    )

    # Extract only if it is a ZIP by content; otherwise leave as-is
    if zipfile.is_zipfile(out_path):
        try:
            files = _extract_top_level_zip(out_path, out_dir)
        except Exception:
            # keep the archive if something goes wrong so you can inspect it
            files = [str(out_path)]
    else:
        files = [str(out_path)]

    return {"ok": True, 'N.º Procedimento (ID BASE)': row['N.º Procedimento (ID BASE)'], "files": files}



# para dar unzip de pastas que estão zip dentro da pasta extraída


SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe" # tive de fazer download disto
def _ensure_7z_available():
    exe = shutil.which(SEVEN_ZIP)
    if exe is None:
        raise FileNotFoundError(
            f"7-Zip not found. Set SEVEN_ZIP to full path, e.g. r'C:\\Program Files\\7-Zip\\7z.exe', "
            f"or add 7-Zip to PATH."
        )
    return exe

def _extract_zip_flat(zip_path: Path, out_dir: Path,
                      max_depth: int = 3, depth: int = 0,
                      max_files: int = 5000,
                      max_uncompressed_bytes: int = 2_000_000_000) -> list[str]:
    saved = []
    file_count = 0
    total_uncompressed = 0

    def _extract_stream(zf: zipfile.ZipFile, out_dir: Path, depth: int):
        nonlocal file_count, total_uncompressed, saved
        for info in zf.infolist():
            if info.is_dir():
                continue
            total_uncompressed += info.file_size
            file_count += 1
            if file_count > max_files:
                raise RuntimeError("Too many files in nested zips.")
            if total_uncompressed > max_uncompressed_bytes:
                raise RuntimeError("Uncompressed size limit exceeded.")

            name = _safe_name(Path(info.filename).name)
            dest = _unique_path(out_dir, name)

            with zf.open(info, "r") as src:
                data = src.read()

            # nested zip?
            if name.lower().endswith(".zip") and depth < max_depth:
                try:
                    with zipfile.ZipFile(io.BytesIO(data), "r") as nested:
                        _extract_stream(nested, out_dir, depth + 1)
                    continue  # don't save the intermediate .zip
                except zipfile.BadZipFile:
                    pass

            with open(dest, "wb") as f:
                f.write(data)
            saved.append(str(dest))

    with zipfile.ZipFile(zip_path, "r") as zf:
        _extract_stream(zf, out_dir, depth)

    try:
        zip_path.unlink()
    except Exception:
        pass
    return saved

def _extract_with_7z(archive_path: Path, out_dir: Path) -> list[str]:
    exe = _ensure_7z_available()
    tmp = Path(tempfile.mkdtemp(prefix="unpack_7z_"))
    try:
        cmd = [exe, "x", "-y", str(archive_path), f"-o{tmp}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())

        saved = []
        for p in tmp.rglob("*"):
            if p.is_file():
                dest = _unique_path(out_dir, _safe_name(p.name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                saved.append(str(dest))

        try:
            archive_path.unlink()
        except Exception:
            pass

        return saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def extract_archives_under(root: Path, max_zip_depth: int = 3, recursive: bool = False):
    """
    Extracts .zip (recursively) and .rar inside each contract folder under `root`.
    Deletes archives after successful extraction.
    Avoids infinite loops by tracking failures and breaking on no-progress.
    """
    root = Path(root)
    failed: set[Path] = set()
    summary = {"processed": [], "errors": []}

    # Choose iterator (only top-level files vs fully recursive)
    def find_archives(contract_dir: Path):
        it = contract_dir.rglob("*") if recursive else contract_dir.iterdir()
        return [p for p in it if p.is_file() and p.suffix.lower() in {".zip", ".rar",  ".7z"} and p not in failed]

    for contract_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # keep extracting until we stop making progress
        while True:
            archives = find_archives(contract_dir)
            if not archives:
                break

            progress = False
            for arc in archives:
                try:
                    if arc.suffix.lower() == ".zip" and zipfile.is_zipfile(arc):
                        files = _extract_zip_flat(arc, contract_dir, max_depth=max_zip_depth)
                    else:
                        # RAR or weird ZIP -> 7z
                        files = _extract_with_7z(arc, contract_dir)

                    summary["processed"].append({
                        "contract_dir": str(contract_dir),
                        "archive": str(arc),
                        "extracted_files": files
                    })
                    progress = True
                except Exception as e:
                    failed.add(arc)  # don't retry this archive again
                    summary["errors"].append({
                        "contract_dir": str(contract_dir),
                        "archive": str(arc),
                        "error": repr(e)
                    })
                    print(f"Failed to extract {arc}: {e}")

            if not progress:
                # No archives were extracted this pass -> avoid infinite loop
                break

    return summary


# keep only important docs


def _norm(s: str) -> str:
    # lower + strip accents + compress spaces
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- pattern matchers (filename or text) ---
# Matches:
# - "caderno" or "cadernos"
# - "caderno de encargos" / "cadernos de encargos" (with optional "de")
_CADERNO_RX = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])caderno(?:s)?(?![A-Za-z0-9])"                        # standalone "caderno(s)"
    r"|"                                                                    # or
    r"(?<![A-Za-z0-9])caderno(?:s)?(?:[\s_]+(?:de[\s_]+)?encargo(?:s)?)(?![A-Za-z0-9])"  # caderno de encargos
    r"|"                                                                    # or CE between underscores / edges
    r"(?:(?:^|_)ce(?:_|$))"
    r")"
    
)

# Matches:
# - "programa" or "programas" (standalone)
# - optional tail: "do procedimento" or "do concurso"
_PROGRAMA_RX = re.compile(
    r"(?<![A-Za-z0-9])programa(?:s)?(?:[\s_]+(?:do[\s_]+)?(?:procedimento|concurso))?(?![A-Za-z0-9])"
)

# Optional: recognize contracts too (both EN and PT)
_CONTRATO_RX = re.compile(
    r"\bcontract|\bcontrato(?:s)?\b"
)

def _classify_by_text(norm_text: str) -> Optional[Literal["caderno","programa","contrato"]]:
    # If you want "contrato" to win whenever present, keep this first
    if _CONTRATO_RX.search(norm_text):
        return "contrato"
    if _CADERNO_RX.search(norm_text):
        return "caderno"
    if _PROGRAMA_RX.search(norm_text):
        return "programa"
    return None



def _classify_file(path: Path) -> Optional[Literal["caderno","programa"]]:
    """
    Decide if a file is 'caderno' or 'programa' based on:
      1) filename (fast)
      2) content text (fallback)
    Returns None if not recognized.
    """
    # skip containers and junk
    if path.suffix.lower() in {".zip", ".rar", ".7z"}:
        return None

    # 1) filename heuristic
    base_norm = _norm(path.name)
    typ = _classify_by_text(base_norm)
    if typ:
        return typ

    return None



def keep_only_key_docs(folder: Path, delete_nonimportant: bool = False):
    """
    Scans 'folder', classifies files as:
      - 'caderno' (Caderno de Encargos)
      - 'programa' (Programa do Procedimento)
      - None (irrelevant/unknown)
    If delete_nonimportant=True, deletes unclassified files.

    Returns a dict summary.
    """
    folder = Path(folder)
    kept, removed, errors = [], [], []

    for entry in sorted(folder.iterdir()):
        if entry.is_dir():
            # keep subfolders as-is; or handle recursively if you prefer
            continue
        try:
            kind = _classify_file(entry)
        except Exception as e:
            errors.append({"file": str(entry), "error": repr(e)})
            kind = None

        if kind is not None:
            kept.append({"file": str(entry), "type": kind})
        else:
            if delete_nonimportant:
                try:
                    entry.unlink()
                    removed.append(str(entry))
                except Exception as e:
                    errors.append({"file": str(entry), "error": f"unlink failed: {e}"})

    return {
        "folder": str(folder),
        "kept": kept,               # [{file, type}]
        "removed": removed,         # [file,...] (only if delete_nonimportant=True)
        "errors": errors
    }


# pdf do anuncio em conjunto com o contrato



URL_BASE = "https://www.base.gov.pt/Base4/pt/resultados/?type=doc_documentos&id={doc_id}&ext=.pdf"

# --- session (reused) ---
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.base.gov.pt/",
    "Accept": "application/pdf,*/*",
})

# --- small helpers ---
def _to_list(v):
    if v is None: return []
    if isinstance(v, float) and math.isnan(v): return []
    if isinstance(v, (list, tuple, set)): return list(v)
    return [v]

def _safe_name(s: str) -> str:
    s = (s or "").strip() or "file"
    return re.sub(r'[\\/*?:"<>|]+', "_", s)

def _unique_path(base_dir: Path, name: str) -> Path:
    p = base_dir / name
    if not p.exists(): return p
    stem, ext = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        q = base_dir / f"{stem} ({i}){ext}"
        if not q.exists(): return q
        i += 1


def _pick_pdf_filename(resp, url_tail: str, fallback: str) -> str:
    # always use the fallback the caller provides
    name = fallback if fallback.lower().endswith(".pdf") else f"{fallback}.pdf"
    return _safe_name(name)

# --- unified downloader ---
def download_contract_pdfs(row,
                           out_folder: str | Path = "../docs",
                           id_col="contract_id",
                           anuncio_col="anuncio_pdf",
                           ids_col="ids_extraidos_docs",
                           url_base: str = URL_BASE,
                           connect_timeout=10,
                           read_timeout=60,
                           chunk_size=64*1024):
    """
    Saves PDFs for one contract:
      - If row[anuncio_col] is a URL -> download it
      - For each id in row[ids_col] -> download URL_BASE.format(doc_id=id)
    Files go to <out_folder>/<contract_id>/ . Returns a summary dict.
    """
    contract_id = str(row.get(id_col) or row.get("id"))
    out_dir = Path(out_folder) / contract_id
    created_dir = False
    saved, skipped = [], []

    # Build a worklist: [{'url': ..., 'hint': ...}, ...]
    work = []
    anuncio_url = row.get(anuncio_col)
    if isinstance(anuncio_url, str) and anuncio_url.strip().startswith(("http://", "https://")):
        work.append({"url": anuncio_url.strip(), "hint": "anuncio"})

    for doc_id in _to_list(row.get(ids_col)):
        work.append({"url": url_base.format(doc_id=doc_id), "hint": f"contract", "doc_id": doc_id})

    for item in work:
        url = item["url"]
        hint = item["hint"]
        try:
            r = SESSION.get(url, stream=True, timeout=(connect_timeout, read_timeout), allow_redirects=True)
            status = r.status_code

            # quick HTTP checks
            if status == 404:
                skipped.append({"url": url, "reason": "404"})
                r.close(); continue
            if status >= 400:
                skipped.append({"url": url, "reason": f"HTTP {status}"})
                r.close(); continue

            # peek first chunk
            it = r.iter_content(chunk_size)
            try:
                first = next(it)
            except StopIteration:
                skipped.append({"url": url, "reason": "empty response"})
                r.close(); continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            is_pdf = first.startswith(b"%PDF-") or ("pdf" in ctype)
            if not is_pdf:
                skipped.append({"url": url, "reason": f"not pdf (Content-Type={ctype or 'unknown'})"})
                r.close(); continue

            # ensure directory only when we know it's valid
            if not created_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                created_dir = True

            if hint == "anuncio":
                fallback_name = f"anuncio_{contract_id}.pdf"
            else:
                doc_id = item["doc_id"]
                fallback_name = f"contract_{doc_id}.pdf"   # <- type _ doc_id, as requested
            

            # choose filename (header -> URL -> hint)
            fname = _pick_pdf_filename(r, urlparse(url).path, fallback_name)
            dest = _unique_path(out_dir, fname)

            # write first chunk + rest
            with open(dest, "wb") as f:
                f.write(first)
                for chunk in it:
                    if chunk:
                        f.write(chunk)

            saved.append(str(dest))
        except requests.RequestException as e:
            skipped.append({"url": url, "reason": e.__class__.__name__})
            continue

    return {
        "contract_id": contract_id,
        "created_folder": created_dir,
        "saved": saved,
        "skipped": skipped
    }

from typing import List
def get_covid_legal_frameworks() -> List[str]:
    return ["10-A/2020, de 13.03",
        "1-A/2020, de 20.03",
        "30/2021, de 21.05"]