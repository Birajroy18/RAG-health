import os
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any

import requests

from load_secrets import init_secrets

init_secrets()

# --- Config ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Avoid openrouter/free — it may route to reasoning-only models that return empty content.
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free"
).strip() or "meta-llama/llama-3.2-3b-instruct:free"
OPENROUTER_FALLBACK_MODELS = os.getenv(
    "OPENROUTER_FALLBACK_MODELS",
    "meta-llama/llama-3.3-70b-instruct:free,openrouter/free",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
CDC_MEDIA_URL = "https://tools.cdc.gov/api/v2/resources/media"

PUBMED_MAX_RESULTS = 6
EUROPE_PMC_MAX_RESULTS = 6
OPENFDA_MAX_RESULTS = 2
MEDLINEPLUS_MAX_RESULTS = 3
CDC_MAX_RESULTS = 2
MAX_SOURCES_TO_LLM = 5
MAX_SOURCE_TEXT_CHARS = 800
LLM_MAX_TOKENS = 1600
REQUEST_TIMEOUT = 25
MIN_CONTEXT_CHARS = 280
MIN_RELEVANCE_SCORE = 0.12
MAX_OPENROUTER_RETRY_WAIT_SECONDS = 20

NO_ANSWER_MESSAGE = (
    "No related data found in the trusted medical sources. "
    "Try rephrasing your question with more specific medical terms."
)

OPENROUTER_BUSY_MESSAGE = (
    "The medical sources were found, but the free OpenRouter model is temporarily "
    "rate-limited. Please try again in a minute, or add your own OpenRouter provider "
    "key/rate limits in OpenRouter settings."
)

LATEST_KEYWORDS = re.compile(
    r"\b(latest|recent|newest|current|updated|202[4-9]|20[3-9][0-9])\b",
    re.IGNORECASE,
)
RESEARCH_KEYWORDS = re.compile(
    r"\b(research|studies|study|clinical trials|what do studies show|what studies show|"
    r"what does research show|new research|new studies)\b",
    re.IGNORECASE,
)
INTERACTION_KEYWORDS = re.compile(r"\b(interact|interaction|interacting)\b", re.IGNORECASE)
PATIENT_EDUCATION_KEYWORDS = re.compile(
    r"\b(symptom|symptoms|sign|signs|warning|early|treatment|treat|steps|"
    r"management|manage|prevention|prevent|when to seek|emergency|urgent|"
    r"disease|condition|cure|measures|cause|causes|caused by|risk|risks|"
    r"side\s*effect|side\s*effects|headache|runny|running|nose|cold|flu|"
    r"cough|fever|sore throat|symptomps|chicken\s*pox|chickenpox|varicella)\b",
    re.IGNORECASE,
)
DRUG_KEYWORDS = re.compile(
    r"\b(drug|medication|medicine|pill|tablet|capsule|dose|dosage|mg|ml|"
    r"prescription|side\s*effect|contraindicat|interact|fda|approved|"
    r"metformin|insulin|atorvastatin|amlodipine|lisinopril|warfarin)\b",
    re.IGNORECASE,
)
OPENFDA_STOPWORDS = {
    "about", "approved", "can", "cause", "caused", "causes", "could", "does",
    "drug", "drugs", "effect", "effects", "from", "how", "medicine",
    "medication", "side", "symptom", "symptoms", "tablet", "treatment",
    "what", "when", "which", "with", "and", "or", "without", "supplement",
    "supplements", "interaction", "interact", "interacting",
}
KNOWN_DRUG_TERMS = {
    "amlodipine",
    "atorvastatin",
    "aspirin",
    "acetaminophen",
    "ibuprofen",
    "insulin",
    "lisinopril",
    "metformin",
    "paracetamol",
    "warfarin",
}
COMMON_MEDICAL_TERMS = KNOWN_DRUG_TERMS.union(
    {
        "hypertension",
        "diabetes",
        "chickenpox",
        "varicella",
        "lactic acidosis",
        "migraine",
        "headache",
        "runny nose",
        "upper respiratory infection",
        "common cold",
        "flu",
    }
)
COMMON_SPELLING_FIXES = {
    "peracitamol": "paracetamol",
    "paracetmol": "paracetamol",
    "paracitamol": "paracetamol",
    "paracatemol": "paracetamol",
    "paracetemol": "paracetamol",
}

_last_ncbi_request_at = 0.0


class OpenRouterBusyError(RuntimeError):
    """Raised when OpenRouter or its upstream free providers are temporarily busy."""


class GeminiError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


def _ncbi_params() -> dict[str, str]:
    params: dict[str, str] = {
        "tool": os.getenv("NCBI_TOOL", "medical-rag-assistant"),
        "email": os.getenv("NCBI_EMAIL", "user@example.com"),
    }
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _throttle_ncbi() -> None:
    """NCBI: max ~3 requests/sec without API key."""
    global _last_ncbi_request_at
    min_interval = 0.34 if not os.getenv("NCBI_API_KEY") else 0.11
    elapsed = time.time() - _last_ncbi_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_ncbi_request_at = time.time()


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    stop = {
        "the", "and", "for", "are", "what", "how", "can", "does", "with",
        "from", "that", "this", "about", "have", "has", "was", "were", "not",
    }
    return {t for t in tokens if t not in stop}


def _overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    t = _tokenize(text)
    return len(q & t) / len(q)


def _http_get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def _wants_latest(query: str) -> bool:
    return bool(LATEST_KEYWORDS.search(query))


def _wants_research(query: str) -> bool:
    return bool(RESEARCH_KEYWORDS.search(query))


def _wants_drug_interaction(query: str) -> bool:
    return bool(INTERACTION_KEYWORDS.search(query) and DRUG_KEYWORDS.search(query))


def _wants_drug_sources(query: str) -> bool:
    return bool(DRUG_KEYWORDS.search(query) or _wants_drug_interaction(query))


def _wants_patient_education(query: str) -> bool:
    return bool(
        PATIENT_EDUCATION_KEYWORDS.search(query)
        and not (_wants_latest(query) or _wants_research(query) or _wants_drug_interaction(query))
    )


def _strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = unescape(text)
    return " ".join(text.split())


def _correct_spelling(query: str) -> str:
    def replace_word(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in COMMON_MEDICAL_TERMS or len(lower) <= 3:
            return word
        if lower in COMMON_SPELLING_FIXES:
            return COMMON_SPELLING_FIXES[lower]
        best_match = difflib.get_close_matches(lower, COMMON_MEDICAL_TERMS, n=1, cutoff=0.7)
        return best_match[0] if best_match else word

    return re.sub(r"\b[a-z]{4,}\b", replace_word, query, flags=re.IGNORECASE)


def _normalize_query(query: str) -> str:
    replacements = {
        r"\bruning\s+nose\b": "runny nose",
        r"\brunning\s+nose\b": "runny nose",
        r"\bsymptomps\b": "symptoms",
        r"\bchicken\s+pox\b": "chickenpox",
    }
    normalized = query.lower()
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = _correct_spelling(normalized)
    return normalized


def _simplify_patient_education_query(query: str) -> str:
    lower = _normalize_query(query)
    medical_terms = ["fever", "headache", "cough", "cold", "flu", "chickenpox", "hypertension", "diabetes"]
    for term in medical_terms:
        if term in lower:
            if re.search(r"\b(treat|treatment|manage|management|care|options|how do i|how to)\b", lower):
                return f"{term} treatment"
            if re.search(r"\b(symptom|symptoms|sign|signs)\b", lower):
                return f"{term} symptoms"
            if re.search(r"\b(cause|causes|caused by|why)\b", lower):
                return f"{term} causes"
            return term
    return lower


def _consumer_search_terms(query: str) -> list[str]:
    lower = _normalize_query(query)
    terms: list[str] = []
    has_headache = "headache" in lower or "head ache" in lower
    has_runny_nose = "runny nose" in lower

    if "fever" in lower:
        if re.search(r"\b(treat|treatment|manage|management|care|options|how do i|how to)\b", lower):
            terms.extend(["fever treatment", "fever", "fever management", "fever care"])
        elif re.search(r"\b(symptom|symptoms|sign|signs)\b", lower):
            terms.extend(["fever symptoms", "fever", "fever signs"])
        elif re.search(r"\b(cause|causes|caused by|why)\b", lower):
            terms.extend(["fever causes", "fever"])
        else:
            terms.append("fever")

    if "stroke" in lower:
        if re.search(r"\b(sign|signs|symptom|symptoms|warning|early)\b", lower):
            terms.extend(["stroke signs symptoms", "stroke"])
        else:
            terms.append("stroke")
    if "hypertension" in lower or "high blood pressure" in lower:
        if re.search(r"\b(treat|treatment|manage|management|steps)\b", lower):
            terms.extend(["high blood pressure treatment", "high blood pressure"])
        else:
            terms.extend(["high blood pressure", "hypertension"])
    if "type 2 diabetes" in lower or "diabetes type 2" in lower:
        terms.extend(["type 2 diabetes", "diabetes"])
    if "lactic acidosis" in lower:
        terms.append("lactic acidosis")
    if "chickenpox" in lower or "varicella" in lower:
        terms.extend(["chickenpox symptoms", "chickenpox", "varicella"])
    if has_headache and has_runny_nose:
        terms.extend([
            "headache runny nose",
            "common cold treatment",
            "flu symptoms",
            "sinusitis symptoms",
            "sinusitis treatment",
            "upper respiratory symptoms",
            "allergic rhinitis symptoms",
            "nasal symptoms headache",
        ])
    elif has_runny_nose:
        terms.extend(["runny nose", "nasal symptoms"])
    elif has_headache:
        terms.extend(["headache", "migraine"])

    cleaned = " ".join(re.findall(r"[a-z0-9][a-z0-9-]{2,}", lower))
    if cleaned:
        terms.append(cleaned)

    return list(dict.fromkeys(terms))


def _primary_topic_tokens(query: str) -> set[str]:
    lower = _normalize_query(query)
    if "stroke" in lower:
        return {"stroke"}
    if "hypertension" in lower or "high blood pressure" in lower:
        return {"hypertension", "blood", "pressure"}
    if "type 2 diabetes" in lower or "diabetes type 2" in lower:
        return {"diabetes"}
    if "metformin" in lower:
        return {"metformin"}
    if "chickenpox" in lower or "varicella" in lower:
        return {"chickenpox", "varicella"}
    if "headache" in lower and "runny nose" in lower:
        return set()
    if "runny nose" in lower:
        return set()
    if "headache" in lower:
        return {"headache", "migraine"}
    return set()


def _build_literature_queries(query: str) -> tuple[str, str]:
    """Build PubMed and Europe PMC query strings (optional journal filters)."""
    pubmed_term = query
    epmc_term = query

    if re.search(r"\bnature\b", query, re.IGNORECASE):
        pubmed_term = f'({query}) AND "Nature"[Journal]'
        epmc_term = f'({query}) JOURNAL:"Nature"'
    elif re.search(r"\blancet\b", query, re.IGNORECASE):
        pubmed_term = f'({query}) AND "Lancet"[Journal]'
        epmc_term = f'({query}) JOURNAL:"Lancet"'

    return pubmed_term, epmc_term


def _europe_pmc_article_url(pmid: str, pmcid: str, doi: str, source: str, record_id: str) -> str:
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"
    if pmcid:
        pmcid_num = pmcid.replace("PMC", "", 1)
        return f"https://europepmc.org/article/PMC/{pmcid_num}"
    if doi:
        return f"https://doi.org/{doi}"
    if source and record_id:
        return f"https://europepmc.org/article/{source}/{record_id}"
    return "https://europepmc.org/"


def search_pubmed(query: str) -> list[dict]:
    """Search PubMed and return article records with abstracts."""
    pubmed_term, _ = _build_literature_queries(query)
    sort = "pub+date" if _wants_latest(query) else "relevance"
    search_params = {
        "db": "pubmed",
        "term": pubmed_term,
        "retmax": PUBMED_MAX_RESULTS,
        "retmode": "json",
        "sort": sort,
        **_ncbi_params(),
    }

    _throttle_ncbi()
    search_resp = _http_get(f"{NCBI_EUTILS_BASE}/esearch.fcgi", search_params)
    id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "rettype": "xml",
        "retmode": "xml",
        **_ncbi_params(),
    }
    _throttle_ncbi()
    fetch_resp = _http_get(f"{NCBI_EUTILS_BASE}/efetch.fcgi", fetch_params)
    return _parse_pubmed_xml(fetch_resp.text, id_list)


def _parse_pubmed_xml(xml_text: str, id_order: list[str]) -> list[dict]:
    """Parse PubMed XML into source dicts, preserving esearch rank."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    articles: dict[str, dict] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

        abstract_parts = []
        for abs_el in article.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            part = "".join(abs_el.itertext()).strip()
            if part:
                abstract_parts.append(f"{label}: {part}" if label else part)
        abstract = " ".join(abstract_parts).strip()

        year = ""
        year_el = article.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            year = year_el.text.strip()

        journal_el = article.find(".//Title")
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""

        text = f"Title: {title}\n"
        if journal:
            text += f"Journal: {journal}\n"
        if year:
            text += f"Year: {year}\n"
        if abstract:
            text += f"Abstract: {abstract}\n"
        else:
            text += "Abstract: (not available in PubMed record)\n"

        articles[pmid] = {
            "source": "PubMed",
            "source_id": pmid,
            "pmid": pmid,
            "title": title,
            "text": text.strip(),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "published": year,
        }

    ordered = []
    for rank, pmid in enumerate(id_order, start=1):
        if pmid in articles:
            item = articles[pmid].copy()
            item["rank"] = rank
            ordered.append(item)
    return ordered


def _clean_openfda_terms(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", query.lower())
    return [word for word in words if word not in OPENFDA_STOPWORDS]


def _build_openfda_search_expr(query: str, interaction: bool = False) -> str:
    terms = _clean_openfda_terms(query)
    if not terms:
        return ""

    drug = next((term for term in terms if term in KNOWN_DRUG_TERMS), terms[0])
    partner = None
    if interaction:
        partner = next(
            (term for term in terms if term != drug and term not in {"interaction", "interact", "interacting"}),
            None,
        )

    if interaction and partner:
        phrase = f"{drug} {partner} interaction"
    elif interaction:
        phrase = f"{drug} interaction"
    else:
        phrase = " ".join(terms[:4])

    parts = [
        f'openfda.generic_name:"{drug}"',
        f'openfda.brand_name:"{drug}"',
        f'active_ingredient:"{drug}"',
    ]

    if interaction and partner:
        parts.extend(
            [
                f'drug_interactions:"{partner}"',
                f'warnings_and_precautions:"{phrase}"',
                f'contraindications:"{phrase}"',
            ]
        )
    elif len(terms) > 1:
        parts.extend(
            [
                f'warnings_and_precautions:"{phrase}"',
                f'boxed_warning:"{phrase}"',
                f'adverse_reactions:"{phrase}"',
            ]
        )

    return " OR ".join(parts)


def _promote_general_medlineplus_topic(sources: list[dict], query: str) -> list[dict]:
    lower = _normalize_query(query)
    general_promotions = []
    if "fever" in lower:
        general_promotions.append("fever")
    if "headache" in lower:
        general_promotions.append("headache")
    if "hypertension" in lower or "high blood pressure" in lower:
        general_promotions.append("high blood pressure")
    if "diabetes" in lower:
        general_promotions.append("diabetes")

    for promotion in general_promotions:
        for index, source in enumerate(sources):
            title = source.get("title", "").strip().lower()
            if title == promotion:
                sources.insert(0, sources.pop(index))
                return sources
    return sources


def search_medlineplus(query: str) -> list[dict]:
    """Search MedlinePlus health topics (free NLM API, no key required)."""
    sources = []
    for term in _consumer_search_terms(query)[:6]:
        params = {
            "db": "healthTopics",
            "term": term,
            "retmax": MEDLINEPLUS_MAX_RESULTS,
            "rettype": "brief",
            "tool": os.getenv("NCBI_TOOL", "medical-rag-assistant"),
            "email": os.getenv("NCBI_EMAIL", "user@example.com"),
        }
        try:
            response = _http_get(MEDLINEPLUS_SEARCH_URL, params)
        except requests.RequestException:
            continue

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            continue

        for doc in root.findall(".//document"):
            url = (doc.get("url") or "").strip()
            if not url:
                continue

            content = {
                (node.get("name") or ""): _strip_markup("".join(node.itertext()))
                for node in doc.findall("content")
            }
            title = content.get("title") or "MedlinePlus health topic"
            summary = content.get("fullSummary") or content.get("snippet") or ""
            if not summary:
                continue

            sources.append(
                {
                    "source": "MedlinePlus",
                    "source_id": url.rsplit("/", 1)[-1] or title,
                    "title": title,
                    "text": f"Title: {title}\nSummary: {summary}",
                    "url": url,
                    "published": "",
                    "rank": int(doc.get("rank") or len(sources) + 1) + 1,
                }
            )

    sources = _dedupe_sources(sources)
    sources = _promote_general_medlineplus_topic(sources, query)
    for rank, item in enumerate(sources, start=1):
        item["rank"] = rank
    return sources


def search_cdc(query: str) -> list[dict]:
    """Search CDC syndicated pages by title and fetch short content when available."""
    sources = []
    for term in _consumer_search_terms(query)[:6]:
        params = {
            "title": term,
            "max": CDC_MAX_RESULTS,
            "mediatype": "HTML",
            "languagename": "English",
        }
        try:
            response = _http_get(CDC_MEDIA_URL, params)
        except requests.RequestException:
            continue

        payload = response.json()
        if payload.get("meta", {}).get("status") != 200:
            continue

        for rank, item in enumerate(payload.get("results", []) or [], start=1):
            title = (item.get("name") or "").strip()
            description = _strip_markup(item.get("description") or item.get("subTitle") or "")
            url = item.get("sourceUrl") or item.get("targetUrl") or item.get("persistentUrl") or ""
            content_url = item.get("contentUrl") or ""
            body = description

            if content_url:
                try:
                    content_resp = _http_get(content_url)
                    body = _strip_markup(content_resp.text)[:2500] or body
                except requests.RequestException:
                    pass

            if not title or not url or not body:
                continue

            sources.append(
                {
                    "source": "CDC",
                    "source_id": str(item.get("id") or title),
                    "title": title,
                    "text": f"Title: {title}\nSummary: {body}",
                    "url": url,
                    "published": item.get("datePublished") or item.get("dateModified") or "",
                    "rank": rank,
                }
            )

    return _dedupe_sources(sources)


def search_openfda(query: str, interaction: bool = False) -> list[dict]:
    """Search OpenFDA drug labels (free API, no key required)."""
    search_expr = _build_openfda_search_expr(query, interaction=interaction)
    if not search_expr:
        return []

    params = {"search": search_expr, "limit": OPENFDA_MAX_RESULTS}

    try:
        response = _http_get(OPENFDA_LABEL_URL, params)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in {400, 404}:
            return []
        raise

    results = response.json().get("results", [])
    sources = []
    for rank, record in enumerate(results, start=1):
        openfda = record.get("openfda", {}) or {}
        brand_names = openfda.get("brand_name", []) or []
        generic_names = openfda.get("generic_name", []) or []
        brand = brand_names[0] if brand_names else "Unknown brand"
        generic = generic_names[0] if generic_names else ""

        sections = []
        for field, label in (
            ("indications_and_usage", "Indications and usage"),
            ("warnings_and_precautions", "Warnings"),
            ("contraindications", "Contraindications"),
            ("drug_interactions", "Drug interactions"),
            ("dosage_and_administration", "Dosage"),
        ):
            values = record.get(field)
            if values:
                body = values[0] if isinstance(values, list) else str(values)
                sections.append(f"{label}: {body[:2000]}")

        if not sections:
            continue

        text = f"Drug (brand): {brand}\n"
        if generic:
            text += f"Generic name: {generic}\n"
        text += "\n".join(sections)

        sources.append(
            {
                "source": "OpenFDA",
                "source_id": brand,
                "title": f"{brand} ({generic})" if generic else brand,
                "text": text.strip(),
                "url": "https://open.fda.gov/apis/drug/label/",
                "published": "",
                "rank": rank,
            }
        )
    return sources


def search_europe_pmc(query: str) -> list[dict]:
    """Search Europe PMC (free REST API). Adds OA metadata and Lancet/Nature coverage."""
    _, epmc_term = _build_literature_queries(query)
    params: dict[str, Any] = {
        "query": epmc_term,
        "format": "json",
        "pageSize": EUROPE_PMC_MAX_RESULTS,
        "resultType": "core",
    }
    if _wants_latest(query):
        params["sort"] = "PUBLISHED_DATE desc"

    try:
        response = _http_get(EUROPE_PMC_SEARCH_URL, params)
        results = response.json().get("resultList", {}).get("result", []) or []
    except requests.RequestException:
        return []

    sources = []
    for rank, record in enumerate(results, start=1):
        title = (record.get("title") or "").strip() or "Untitled"
        abstract = (record.get("abstractText") or "").strip()
        journal = (record.get("journalTitle") or "").strip()
        year = str(record.get("pubYear") or "").strip()
        pmid = str(record.get("pmid") or "").strip()
        pmcid = str(record.get("pmcid") or "").strip()
        doi = str(record.get("doi") or "").strip()
        source_db = str(record.get("source") or "").strip()
        record_id = str(record.get("id") or "").strip()
        is_oa = record.get("isOpenAccess") == "Y"
        has_fulltext = record.get("hasText") == "Y"

        text = f"Title: {title}\n"
        if journal:
            text += f"Journal: {journal}\n"
        if year:
            text += f"Year: {year}\n"
        if is_oa:
            text += "Open access: Yes\n"
        if has_fulltext:
            text += "Full text in Europe PMC: Yes\n"
        if abstract:
            text += f"Abstract: {abstract}\n"
        else:
            text += "Abstract: (not available)\n"

        source_id = pmid or pmcid or doi or record_id or f"epmc-{rank}"
        sources.append(
            {
                "source": "Europe PMC",
                "source_id": source_id,
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "text": text.strip(),
                "url": _europe_pmc_article_url(pmid, pmcid, doi, source_db, record_id),
                "published": year,
                "rank": rank,
            }
        )
    return sources


def _merge_literature_sources(pubmed_sources: list[dict], epmc_sources: list[dict]) -> list[dict]:
    """Merge Europe PMC into PubMed results, skipping duplicate PMIDs."""
    merged = list(pubmed_sources)
    seen_pmids = {s["pmid"] for s in pubmed_sources if s.get("pmid")}

    for item in epmc_sources:
        pmid = item.get("pmid")
        if pmid and pmid in seen_pmids:
            continue
        merged.append(item)
        if pmid:
            seen_pmids.add(pmid)
    return merged


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    deduped = []
    seen: dict[tuple, dict] = {}
    for item in sources:
        key = (item.get("source"), item.get("url") or item.get("source_id"))
        if key in seen:
            existing = seen[key]
            text = item.get("text", "")
            if text and text not in existing.get("text", ""):
                existing["text"] = f"{existing.get('text', '')}\n{text}"
            continue
        seen[key] = item
        deduped.append(item)
    return deduped


def _source_score(query: str, item: dict, wants_patient_education: bool, wants_research: bool) -> float:
    expanded_query = f"{query} {' '.join(_consumer_search_terms(query))}"
    score = _overlap_score(expanded_query, f"{item['title']} {item['text']}")
    source = item.get("source")
    title_tokens = _tokenize(item.get("title", ""))
    topic_tokens = _primary_topic_tokens(query)
    title = item.get("title", "").lower()

    if wants_patient_education:
        if source in {"MedlinePlus", "CDC"}:
            score += 0.35
            if topic_tokens and not (topic_tokens & title_tokens):
                score *= 0.25
            elif topic_tokens and title_tokens <= topic_tokens:
                score += 0.25
            if re.search(r"\b(runny|running|runing)\s+nose\b", query.lower()) and "headache" in query.lower():
                if any(term in title for term in ("common cold", "flu", "sinusitis", "allergic rhinitis")):
                    score += 0.25
                elif any(term in title for term in ("mpox", "flaccid", "bird flu", "swine flu")):
                    score *= 0.4
        elif source == "OpenFDA":
            score += 0.15
        elif source in {"PubMed", "Europe PMC"}:
            score -= 0.25
    elif wants_research or _wants_latest(query):
        if source in {"PubMed", "Europe PMC"}:
            score += 0.3
        elif source in {"MedlinePlus", "CDC"}:
            score -= 0.05

    if source in ("PubMed", "Europe PMC") and "(not available" in item["text"]:
        score *= 0.6
    if source == "Europe PMC" and "Open access: Yes" in item["text"]:
        score *= 1.05
    return max(score, 0.0)


RESEARCH_MIN_RELEVANCE_SCORE = 0.06


def fetch_live_sources(query: str) -> list[dict]:
    """Query free medical APIs and return ranked, filtered sources."""
    query = _normalize_query(query)
    wants_research = _wants_research(query)
    wants_patient_education = _wants_patient_education(query)
    # For patient-education queries, use ONLY MedlinePlus per user preference.
    if wants_patient_education:
        medline = search_medlineplus(query)
        if not medline:
            # Try token-level retries (e.g., 'fever', 'cough') to broaden MedlinePlus hits
            tokens = re.findall(r"[a-z0-9][a-z0-9-]{2,}", query.lower())
            agg = []
            for t in tokens[:6]:
                try:
                    agg.extend(search_medlineplus(t) or [])
                except Exception:
                    continue
            medline = _dedupe_sources(agg) if agg else []

        # If MedlinePlus still empty, fall back to CDC syndicated content for patient education
        if not medline:
            try:
                cdc = search_cdc(query)
            except Exception:
                cdc = []
            if cdc:
                combined = _dedupe_sources(cdc)
            else:
                return []
        else:
            combined = _dedupe_sources(medline)
    else:
        # Call all five sources in parallel to reduce latency and improve coverage.
        fda_limit = OPENFDA_MAX_RESULTS if _wants_drug_sources(query) else 1
        results = {
            "medline": [],
            "cdc": [],
            "pubmed": [],
            "epmc": [],
            "fda": [],
        }

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(search_medlineplus, query): "medline",
                ex.submit(search_cdc, query): "cdc",
                ex.submit(search_pubmed, query): "pubmed",
                ex.submit(search_europe_pmc, query): "epmc",
                ex.submit(search_openfda, query, _wants_drug_interaction(query)): "fda",
            }

            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    res = fut.result()
                except Exception:
                    # Network or parsing failures return an empty list for that source
                    res = []
                results[key] = res or []

        consumer_sources = results["medline"] + results["cdc"]
        literature = _merge_literature_sources(results["pubmed"], results["epmc"]) if (results["pubmed"] or results["epmc"]) else []
        fda_sources = results["fda"][:fda_limit]

        combined = _dedupe_sources(consumer_sources + literature + fda_sources)
    if not combined:
        return []

    scored: list[dict] = []
    threshold = (
        0.08 if wants_patient_education else RESEARCH_MIN_RELEVANCE_SCORE if wants_research else MIN_RELEVANCE_SCORE
    )
    for item in combined:
        score = _source_score(query, item, wants_patient_education, wants_research)
        item = item.copy()
        item["relevance_score"] = round(score, 3)
        if score >= threshold:
            scored.append(item)

    scored.sort(key=lambda x: (-x["relevance_score"], x.get("rank", 99)))

    if not scored:
        return []

    total_chars = sum(len(s["text"]) for s in scored)
    if total_chars < MIN_CONTEXT_CHARS:
        return []

    for i, item in enumerate(scored[:MAX_SOURCES_TO_LLM], start=1):
        item["rank"] = i
    return scored[:MAX_SOURCES_TO_LLM]


def _truncate_source_text(text: str, max_chars: int = MAX_SOURCE_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def build_prompt(query: str, sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources, start=1):
        blocks.append(
            f"[{i}] Source: {src['source']} | ID: {src['source_id']} | URL: {src['url']}\n"
            f"Title: {src['title']}\n"
            f"{_truncate_source_text(src['text'])}"
        )
    context = "\n\n---\n\n".join(blocks)

    return f"""You are a medical literature assistant. Answer ONLY using the numbered sources below.

Rules:
- Use ONLY facts explicitly stated in the sources. Do NOT use outside knowledge.
- Cite sources inline like [1], [2] matching the numbers below.
- End with a "References" section listing each cited source as: [n] Title — Source (URL).
- If sources do not contain enough information to answer, respond with EXACTLY:
  "{NO_ANSWER_MESSAGE}"
- Do NOT diagnose, prescribe, or give personal medical advice. State that information is general literature only.
- For symptoms/treatments: summarize only what the sources state; do not invent steps.

=== SOURCES ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""


def _truncate_source_text(text: str, max_chars: int = MAX_SOURCE_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_prompt(query: str, sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources, start=1):
        blocks.append(
            f"[{i}] Source: {src['source']} | ID: {src['source_id']} | URL: {src['url']}\n"
            f"Title: {src['title']}\n"
            f"{_truncate_source_text(src['text'])}"
        )
    context = "\n\n---\n\n".join(blocks)

    return f"""You are a medical assistant for education and literature lookup. Answer ONLY using the numbered sources below.

Rules:
- Use ONLY facts explicitly stated in the sources. Do NOT use outside knowledge.
- Cite sources inline like [1], [2] matching the numbers below.
- Do NOT include a References section. Full source details are shown separately in the app.
- If the sources have no relevant information, respond with EXACTLY:
  "{NO_ANSWER_MESSAGE}"
- If you use the no-data message, do NOT include citations or a References section.
- Do not give partial answers like "the sources do not specify" or "provided sources do not contain"; use the exact no-data message instead.
- Do NOT diagnose, prescribe, or give personal medical advice. State that information is general literature only.
- Make the answer easy to scan: use short sections and bullets, not one long paragraph.
- Keep answers concise: normally 120-220 words.
- For symptom questions: do not claim a diagnosis. If the user asks "what disease", answer with source-supported possible causes, not a definite disease. Do not refuse only because an exact diagnosis cannot be determined.
- If sources support related conditions and self-care measures, list 2-3 possible causes when available. Use sections named "Possible causes", "What you can do", and "Get medical help if".
- In "Possible causes", write "could be associated with" or "may be seen with"; never write "you have".
- For symptoms, warning signs, prevention, or treatment questions: prefer MedlinePlus and CDC sources when present, use plain patient-friendly language, and mention urgent/emergency care only when the sources support it.
- For latest/research questions: use sections named "Recent themes" and "Bottom line". Summarize the retrieved studies by theme and say they are retrieved recent sources, not a complete review.
- For drug questions: use sections named "Short answer", "Higher-risk situations", and "Important note" when the sources support those details.

=== SOURCES ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""


def build_concise_retry_prompt(query: str, sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources[:MAX_SOURCES_TO_LLM], start=1):
        blocks.append(
            f"[{i}] {src['source']} | {src['title']}\n"
            f"{_truncate_source_text(src['text'], 500)}"
        )
    context = "\n\n---\n\n".join(blocks)

    return f"""Answer ONLY from these sources. If they are not enough, answer exactly:
{NO_ANSWER_MESSAGE}

Do not diagnose or prescribe. Do not include references. Keep it under 180 words.
If the user asks "what disease", answer with possible causes supported by the sources, not a definite diagnosis.
For symptom questions, use ALL THREE headings below:
Possible causes
- List 2-3 different source-supported possible causes if available. Use "may be associated with", not "you have".
What you can do
- List source-supported general measures only.
Get medical help if
- List source-supported warning signs only.

Sources:
{context}

Question: {query}
Answer:"""


def _extract_message_text(message: dict) -> str:
    """Parse OpenRouter message content across providers and formats."""
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
            elif item.get("text"):
                parts.append(str(item["text"]))
        joined = "".join(parts).strip()
        if joined:
            return joined

    return ""


def _extract_gemini_text(payload: dict) -> str:
    parts = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _call_gemini_text(prompt: str, api_key: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if not api_key:
        raise GeminiError("Gemini API key is missing.")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
        },
    }
    response = requests.post(
        GEMINI_API_URL,
        params={"key": api_key},
        json=payload,
        timeout=60,
    )

    if not response.ok:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = response.text
        raise GeminiError(f"Gemini API error ({response.status_code}): {error_payload}")

    text = _extract_gemini_text(response.json())
    if not text:
        raise GeminiError("Gemini returned no answer text.")
    return text


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _looks_truncated(text: str) -> bool:
    stripped = _without_references(text).strip()
    if not stripped:
        return True
    if re.search(r"(\*|-|\d+\.)\s+[A-Za-z][^.!?]{0,120}$", stripped):
        return True
    return stripped[-1] not in ".!?)"


def _answer_needs_retry(query: str, answer: str) -> bool:
    if _looks_truncated(answer):
        return True

    if _wants_patient_education(query):
        lower = answer.lower()
        required = ("possible causes", "what you can do", "get medical help")
        if _asks_for_possible_disease_and_measures(query) and any(
            heading not in lower for heading in required
        ):
            return True

    return False


def _safe_search_fragment(text: str) -> str:
    words = re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
    return " ".join(words[:8])


def build_retrieval_query(query: str, gemini_api_key: str = "") -> str:
    """
    Use Gemini only to extract terms, not diagnose. The output expands retrieval
    but never becomes the final answer.
    """
    query = _normalize_query(query)
    if _wants_patient_education(query):
        return _simplify_patient_education_query(query)
    if not gemini_api_key:
        return query

    prompt = f"""Extract search terms from this health question. Do not diagnose.

Return ONLY JSON with these keys:
- symptoms: array of symptoms explicitly stated by the user
- duration: duration explicitly stated by the user, or empty string
- medications: array of medications explicitly stated by the user
- search_queries: 2 to 4 short symptom-based search queries

Rules:
- Do not say the user has any disease.
- Do not include a diagnosis field.
- Search queries must preserve the user's symptoms and may use generic phrases like "upper respiratory symptoms", but must not assert a condition as fact.

Question: {query}
"""
    try:
        payload = _parse_json_object(_call_gemini_text(prompt, gemini_api_key, max_tokens=400))
    except GeminiError:
        return query

    fragments = [query]
    for key in ("symptoms", "medications", "search_queries"):
        values = payload.get(key, [])
        if isinstance(values, list):
            fragments.extend(_safe_search_fragment(str(value)) for value in values[:5])

    duration = payload.get("duration")
    if isinstance(duration, str) and duration.strip():
        fragments.append(_safe_search_fragment(duration))

    expanded = " ".join(fragment for fragment in fragments if fragment)
    return expanded or query


def _openrouter_models() -> list[str]:
    models: list[str] = []
    if OPENROUTER_MODEL:
        models.append(OPENROUTER_MODEL)
    models.extend(
        model.strip()
        for model in OPENROUTER_FALLBACK_MODELS.split(",")
        if model.strip()
    )
    safe_default = "meta-llama/llama-3.2-3b-instruct:free"
    if safe_default not in [m.lower() for m in models]:
        models.append(safe_default)
    return list(dict.fromkeys(models))


def _retry_after_seconds(response: requests.Response, payload: Any) -> int:
    header_value = response.headers.get("Retry-After", "")
    if header_value.isdigit():
        return int(header_value)

    if isinstance(payload, dict):
        metadata = payload.get("error", {}).get("metadata", {})
        retry_after = metadata.get("retry_after_seconds")
        if isinstance(retry_after, (int, float)):
            return int(retry_after)

    return 0


def _call_openrouter(prompt: str, api_key: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://medical-rag-assistant.streamlit.app",
        "X-Title": "Medical Literature RAG Assistant",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.1,
    }

    for attempt in range(2):
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = response.text

        if response.ok:
            return response_payload

        if response.status_code == 429:
            retry_after = _retry_after_seconds(response, response_payload)
            if attempt == 0 and 0 < retry_after <= MAX_OPENROUTER_RETRY_WAIT_SECONDS:
                time.sleep(retry_after)
                continue
            raise OpenRouterBusyError(OPENROUTER_BUSY_MESSAGE)

        if response.status_code in {500, 502, 503, 504}:
            raise OpenRouterBusyError(OPENROUTER_BUSY_MESSAGE)

        raise ValueError(f"OpenRouter API error ({response.status_code}): {response_payload}")

    raise OpenRouterBusyError(OPENROUTER_BUSY_MESSAGE)


def generate_answer(
    query: str,
    sources: list[dict],
    openrouter_api_key: str = "",
    gemini_api_key: str = "",
    concise: bool = False,
) -> str:
    """Call Gemini when configured, otherwise OpenRouter."""
    if gemini_api_key:
        try:
            answer = _call_gemini_text(
                build_concise_retry_prompt(query, sources[:MAX_SOURCES_TO_LLM])
                if concise
                else build_prompt(query, sources[:MAX_SOURCES_TO_LLM]),
                gemini_api_key,
                max_tokens=LLM_MAX_TOKENS,
            )
            if not concise and _answer_needs_retry(query, answer):
                answer = _call_gemini_text(
                    build_concise_retry_prompt(query, sources),
                    gemini_api_key,
                    max_tokens=900,
                )
            return answer
        except GeminiError:
            if not openrouter_api_key:
                raise

    attempts = [
        sources[:MAX_SOURCES_TO_LLM],
        sources[:3],
        sources[:2],
    ]
    last_model = OPENROUTER_MODEL
    last_finish = ""
    busy_models = set()

    if not openrouter_api_key:
        raise ValueError("Add a Gemini API key or an OpenRouter API key to generate answers.")

    for model in _openrouter_models():
        for subset in attempts:
            if not subset:
                continue
            prompt = (
                build_concise_retry_prompt(query, subset)
                if concise
                else build_prompt(query, subset)
            )
            try:
                data = _call_openrouter(prompt, openrouter_api_key, model)
            except OpenRouterBusyError:
                busy_models.add(model)
                break

            choices = data.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            message = choice.get("message", {})
            text = _extract_message_text(message)
            if text:
                return text

            last_finish = choice.get("finish_reason") or choice.get("native_finish_reason") or ""
            last_model = data.get("model", model)

        if sources:
            if not concise:
                try:
                    data = _call_openrouter(
                        build_concise_retry_prompt(query, sources[:2]),
                        openrouter_api_key,
                        model,
                    )
                except OpenRouterBusyError:
                    busy_models.add(model)
                    break
            else:
                continue

            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                text = _extract_message_text(choice.get("message", {}))
                if text:
                    return text
                last_finish = choice.get("finish_reason") or choice.get("native_finish_reason") or last_finish
                last_model = data.get("model", model)

    if busy_models and len(busy_models) == len(_openrouter_models()):
        raise OpenRouterBusyError(OPENROUTER_BUSY_MESSAGE)

    hint = (
        f"Model `{last_model}` returned no answer text"
        + (f" (finish_reason={last_finish})." if last_finish else ".")
        + " Add `OPENROUTER_MODEL = \"meta-llama/llama-3.2-3b-instruct:free\"` to secrets.toml and try again."
    )
    raise ValueError(hint)


def _without_references(text: str) -> str:
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)
    text = re.sub(r"\u3010\d+\u3011", "", text)
    text = re.split(r"\n\s*References\s*:?", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def _sanitize_answer(text: str) -> str:
    """Clean up empty placeholders, repeated punctuation, and extraneous lines.

    - Collapse repeated commas and remove comma-only fragments.
    - Remove lines where a heading has no useful content after the colon.
    - Strip parenthetical word-count lines often injected by LLMs.
    """
    if not text:
        return text

    # If the model included the exact no-data message anywhere, prefer returning it alone.
    if NO_ANSWER_MESSAGE.lower() in text.lower():
        return NO_ANSWER_MESSAGE

    # Remove trailing partial no-data fragments if an actual answer is present.
    text = re.sub(
        r"\s*No related data found(?: in the trusted medical sources\.)?\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse multiple commas and remove comma-only fragments
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r",\s*\.", ".", text)

    # Remove lines like "Recent themes: ,,." where nothing useful follows the colon
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # drop explicit word-count lines
        if re.match(r"^\(Word count", stripped, re.IGNORECASE):
            continue
        if ":" in line:
            after = line.split(":", 1)[1].strip()
            # if 'after' is empty or contains only punctuation/commas, skip the line
            if not after or re.fullmatch(r"[\s,\.\-–—()~]*", after):
                continue
        # remove in-answer 'Sources (N)' lines
        if re.match(r"^Sources\s*\(|^Sources:\s*", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim whitespace
    return text.strip()


def _is_no_answer_like(text: str) -> bool:
    lower = text.lower()
    incomplete_markers = (
        "provided sources do not",
        "sources do not specify",
        "sources don't specify",
        "sources do not contain",
        "sources don't contain",
        "not enough information",
        "insufficient information",
        "do not provide treatment",
        "does not provide treatment",
        "do not specify any treatment",
        "does not specify any treatment",
    )
    return NO_ANSWER_MESSAGE.lower() in lower or any(
        marker in lower for marker in incomplete_markers
    )


def _patient_education_fallback(query: str, sources: list[dict]) -> str:
    if not _wants_patient_education(query):
        return NO_ANSWER_MESSAGE

    cause_titles = []
    preferred = ("common cold", "flu", "sinusitis", "allergic rhinitis", "chickenpox")
    blocked = ("bird flu", "swine flu", "mpox", "flaccid")
    for src in sources:
        title = src.get("title", "")
        lower = title.lower()
        if any(term in lower for term in blocked):
            continue
        clean_title = _clean_condition_title(title)
        if any(term in lower for term in preferred) and clean_title not in cause_titles:
            cause_titles.append(clean_title)
        if len(cause_titles) >= 3:
            break

    if not cause_titles:
        return NO_ANSWER_MESSAGE

    possible = "\n".join(f"- {title} may be associated with these symptoms." for title in cause_titles)

    combined_text = " ".join(src.get("text", "") for src in sources).lower()
    measures = []
    if "no cure" in combined_text and "common cold" in combined_text:
        measures.append("- For common cold, the retrieved source says there is no cure; care is mainly symptom-focused.")
    if "heat pads" in combined_text:
        measures.append("- Retrieved sinusitis information mentions heat pads on the inflamed area.")
    if "saline nasal sprays" in combined_text:
        measures.append("- For sinusitis-related symptoms, retrieved sources mention saline nasal sprays.")
    if "vaporizers" in combined_text:
        measures.append("- Retrieved sinusitis information mentions vaporizers.")
    if "pain relievers" in combined_text:
        measures.append("- Retrieved sinusitis information mentions pain relievers.")
    if "decongestants" in combined_text:
        measures.append("- Retrieved sinusitis information mentions decongestants.")

    if not measures:
        measures.append("- The retrieved sources support possible causes, but do not provide enough self-care steps in the fetched snippets.")

    warning_signs = []
    if "trouble breathing" in combined_text:
        warning_signs.append("- Get medical help for trouble breathing.")
    if "symptoms that concern you" in combined_text:
        warning_signs.append("- Contact a health care provider if symptoms concern you.")
    if not warning_signs:
        warning_signs.append("- The fetched snippets do not provide specific warning signs.")

    return (
        "Information provided is general literature only. This is not a diagnosis.\n\n"
        "Possible causes\n"
        f"{possible}\n\n"
        "What you can do\n"
        + "\n".join(measures[:4])
        + "\n\nGet medical help if\n"
        + "\n".join(warning_signs[:3])
    )


def _clean_condition_title(title: str) -> str:
    lower = title.lower()
    if "common cold" in lower:
        return "Common cold"
    if "sinusitis" in lower:
        return "Sinusitis"
    if "allergic rhinitis" in lower:
        return "Allergic rhinitis"
    if "chickenpox" in lower or "varicella" in lower:
        return "Chickenpox"
    if "flu" in lower:
        return "Flu"
    return title


def _parse_openfda_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([^:]+):\s*(.*)$", line)
        if not match:
            continue
        name = match.group(1).strip().lower()
        value = match.group(2).strip()
        if value:
            sections[name] = value
    return sections


def _interaction_query_terms(query: str) -> tuple[str | None, str | None]:
    terms = _clean_openfda_terms(query)
    drug = next((term for term in terms if term in KNOWN_DRUG_TERMS), None)
    partner = next((term for term in terms if term != drug), None)
    return drug, partner


def _section_summary(text: str, max_sentences: int = 1, max_chars: int = 120) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"^(drug interactions|warnings|contraindications)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-Z]\.[\s-]*", "", text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = sentences[0].strip() if sentences else text
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return summary


def _drug_interaction_fallback(query: str, sources: list[dict]) -> str:
    if not _wants_drug_interaction(query):
        return ""

    openfda_sources = [src for src in sources if src.get("source") == "OpenFDA"]
    if not openfda_sources:
        return ""

    drug, partner = _interaction_query_terms(query)
    interactions: list[str] = []
    warnings: list[str] = []
    contraindications: list[str] = []
    for src in openfda_sources:
        sections = _parse_openfda_sections(src.get("text", ""))
        if "drug interactions" in sections:
            preview = _section_summary(sections["drug interactions"])
            if preview:
                interactions.append(preview)
        if "warnings" in sections:
            preview = _section_summary(sections["warnings"])
            if preview:
                warnings.append(preview)
        if "contraindications" in sections:
            preview = _section_summary(sections["contraindications"])
            if preview:
                contraindications.append(preview)

    if not interactions and not warnings and not contraindications:
        return ""

    lines = ["Short answer: This query appears to involve drug interaction label data from OpenFDA."]
    if drug and partner:
        lines.append(f"{drug.title()} and {partner.title()} interaction information is available in the retrieved drug label(s).")
    elif drug:
        lines.append(f"{drug.title()} interaction information is available in the retrieved drug label(s).")

    if interactions:
        lines.append("Higher-risk situations:")
        lines.extend(f"- {item}" for item in interactions[:3])
    elif warnings:
        lines.append("Higher-risk situations:")
        lines.extend(f"- {item}" for item in warnings[:3])

    if contraindications:
        lines.append("Important note:")
        lines.extend(f"- {item}" for item in contraindications[:3])
    else:
        lines.append("Important note: This summary is based on OpenFDA label content and is not medical advice.")

    return _sanitize_answer("\n".join(lines))


def _asks_for_possible_disease_and_measures(query: str) -> bool:
    lower = _normalize_query(query)
    asks_condition = any(term in lower for term in ("what is the disease", "which disease", "what disease"))
    asks_measures = any(
        term in lower
        for term in ("measures", "what can i do", "taken to stop", "cure", "cause", "causes", "caused by")
    )
    return asks_condition or asks_measures


def _chickenpox_symptom_fallback(query: str, sources: list[dict]) -> str:
    lower = _normalize_query(query)
    if "chickenpox" not in lower or "symptoms" not in lower:
        return ""

    combined_text = " ".join(src.get("text", "") for src in sources).lower()
    symptoms = []
    if "rash" in combined_text:
        symptoms.append("- Rash, often starting on the chest, back, and face before spreading.")
    if "itchy" in combined_text or "pruritic" in combined_text:
        symptoms.append("- Itchy rash.")
    if "fluid-filled blisters" in combined_text or "vesicular" in combined_text:
        symptoms.append("- Fluid-filled blisters that eventually crust or scab.")
    if "fever" in combined_text:
        symptoms.append("- Fever may occur before or with the rash.")
    if "tiredness" in combined_text or "malaise" in combined_text:
        symptoms.append("- Tiredness or malaise may occur.")
    if "loss of appetite" in combined_text:
        symptoms.append("- Loss of appetite may occur.")
    if "headache" in combined_text:
        symptoms.append("- Headache may occur.")

    if not symptoms:
        return ""

    return (
        "Information provided is general literature only. This is not a diagnosis.\n\n"
        "Chickenpox symptoms\n"
        + "\n".join(symptoms[:7])
    )


def rag_query(query: str, openrouter_api_key: str = "", gemini_api_key: str = "") -> dict:
    """
    Live API RAG: MedlinePlus + CDC + PubMed + Europe PMC + OpenFDA per question.
    No local index.
    Returns { query, sources, answer, grounded }.
    """
    clean_query = query.strip()
    retrieval_query = build_retrieval_query(clean_query, gemini_api_key)
    sources = fetch_live_sources(retrieval_query)
    if not sources:
        return {
            "query": query,
            "sources": [],
            "answer": NO_ANSWER_MESSAGE,
            "grounded": False,
        }

    if _wants_drug_interaction(clean_query):
        interaction_answer = _drug_interaction_fallback(clean_query, sources)
        if interaction_answer:
            return {
                "query": query,
                "sources": sources,
                "answer": interaction_answer,
                "grounded": True,
            }

    if _wants_patient_education(clean_query) and _asks_for_possible_disease_and_measures(clean_query):
        fallback = _patient_education_fallback(clean_query, sources)
        if fallback != NO_ANSWER_MESSAGE:
            return {
                "query": query,
                "sources": sources,
                "answer": _sanitize_answer(fallback),
                "grounded": True,
            }

    try:
        answer = generate_answer(clean_query, sources, openrouter_api_key, gemini_api_key)
        # Clean up any empty placeholders or repeated punctuation from the LLM output
        answer = _sanitize_answer(answer)
    except OpenRouterBusyError:
        return {
            "query": query,
            "sources": sources,
            "answer": OPENROUTER_BUSY_MESSAGE,
            "grounded": False,
        }
    except ValueError as exc:
        error_text = str(exc)
        if "returned no answer text" in error_text.lower():
            if _wants_drug_interaction(clean_query):
                interaction_fallback = _drug_interaction_fallback(clean_query, sources)
                if interaction_fallback:
                    return {
                        "query": query,
                        "sources": sources,
                        "answer": interaction_fallback,
                        "grounded": True,
                    }
            if _wants_patient_education(clean_query) and _asks_for_possible_disease_and_measures(clean_query):
                fallback = _patient_education_fallback(clean_query, sources)
                if fallback != NO_ANSWER_MESSAGE:
                    return {
                        "query": query,
                        "sources": sources,
                        "answer": _sanitize_answer(fallback),
                        "grounded": True,
                    }
            if sources:
                try:
                    retry_answer = generate_answer(
                        clean_query,
                        sources,
                        openrouter_api_key,
                        gemini_api_key,
                        concise=True,
                    )
                    retry_answer = _sanitize_answer(retry_answer)
                    if not _is_no_answer_like(retry_answer):
                        return {
                            "query": query,
                            "sources": sources,
                            "answer": _without_references(retry_answer),
                            "grounded": True,
                        }
                except (OpenRouterBusyError, GeminiError, ValueError):
                    pass
            return {
                "query": query,
                "sources": [],
                "answer": NO_ANSWER_MESSAGE,
                "grounded": False,
            }
        raise
    except GeminiError as exc:
        return {
            "query": query,
            "sources": sources,
            "answer": f"Gemini could not generate an answer from the retrieved sources: {exc}",
            "grounded": False,
        }

    if _is_no_answer_like(answer) or _answer_needs_retry(clean_query, answer):
        if _wants_drug_interaction(clean_query):
            interaction_fallback = _drug_interaction_fallback(clean_query, sources)
            if interaction_fallback:
                return {
                    "query": query,
                    "sources": sources,
                    "answer": interaction_fallback,
                    "grounded": True,
                }
        fallback = _patient_education_fallback(clean_query, sources)
        if fallback != NO_ANSWER_MESSAGE:
            return {
                "query": query,
                "sources": sources,
                "answer": _sanitize_answer(fallback),
                "grounded": True,
            }
        if sources:
            try:
                retry_answer = generate_answer(
                    clean_query,
                    sources,
                    openrouter_api_key,
                    gemini_api_key,
                    concise=True,
                )
                retry_answer = _sanitize_answer(retry_answer)
                if not _is_no_answer_like(retry_answer):
                    return {
                        "query": query,
                        "sources": sources,
                        "answer": _without_references(retry_answer),
                        "grounded": True,
                    }
            except (OpenRouterBusyError, GeminiError, ValueError):
                pass
        return {
            "query": query,
            "sources": [],
            "answer": NO_ANSWER_MESSAGE,
            "grounded": False,
        }

    grounded = True

    return {
        "query": query,
        "sources": sources,
        "answer": _sanitize_answer(_without_references(answer)),
        "grounded": grounded,
    }
