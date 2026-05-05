import fitz
import os
import json


def extract_text_from_pdf(pdf_path, paper_id):
    results = []

    try:
        doc = fitz.open(pdf_path)

        if doc.is_encrypted:
            print(f"{pdf_path} şifreli, atlandı.")
            return []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            results.append({
                "paper_id": paper_id,
                "page": page_num + 1,
                "text": text,
                "source_file": os.path.basename(pdf_path)
            })

        return results

    except Exception as e:
        print(f"Hata ({pdf_path}): {e}")
        return []


def process_all_pdfs(input_folder="data/raw", output_folder="data/processed"):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            paper_id = file.replace(".pdf", "")

            print(f"İşleniyor: {file}")

            data = extract_text_from_pdf(pdf_path, paper_id)

            output_file = os.path.join(output_folder, f"{paper_id}.json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_all_pdfs()