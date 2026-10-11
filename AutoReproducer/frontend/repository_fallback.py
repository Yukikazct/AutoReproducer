"""Scope the last-resort repository input to the paper whose lookup failed."""
from src.repository_evidence import normalize_repository_url


def fallback_for_pdf(result, pdf_sha256):
    fallback = ((result or {}).get("data") or {}).get("repository_fallback") or {}
    return fallback if (fallback.get("available") is True and pdf_sha256
                        and fallback.get("pdf_sha256") == pdf_sha256) else None


def repository_widget_key(pdf_sha256):
    return "manual_repository_" + pdf_sha256


def manual_repository(value):
    if not value or not value.strip():
        return "", ""
    normalized = normalize_repository_url(value)
    return (normalized, "") if normalized else ("", "请输入有效的 GitHub 仓库地址，例如 https://github.com/owner/repository")
