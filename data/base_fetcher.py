from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    abstract: str
    pdf_url: str
    authors: list[str]
    published: str
    source: str  # "arxiv" | "core"


class PaperFetcher(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        ...

    @abstractmethod
    def download_pdf(self, paper: PaperMetadata) -> str | None:
        ...
