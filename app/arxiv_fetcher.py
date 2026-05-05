import arxiv
import time


def fetch_papers(query, max_results=5, retries=3):
    """
    Fetch papers from arXiv with retry and rate limiting.
    """

    for attempt in range(retries):
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            results = []

            for result in search.results():
                time.sleep(0.34)  # rate limit (~3 req/sec)

                paper = {
                    "title": result.title,
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id,
                    "pdf_url": result.pdf_url
                }

                results.append(paper)

            return results

        except Exception as e:
            print(f"[Attempt {attempt + 1}] Error: {e}")
            time.sleep(2)  # wait before retry

    print("Failed after retries.")
    return []


if __name__ == "__main__":
    papers = fetch_papers("machine learning", max_results=5)

    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['title']}")