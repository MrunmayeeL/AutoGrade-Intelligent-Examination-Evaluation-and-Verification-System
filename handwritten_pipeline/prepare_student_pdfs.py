import os
import yaml
import shutil

def main():
    metadata_path = os.path.join("temp_extracted", "OS Answer Sheets", "OS_End_Sem_students", "assignment_7244773_export", "submission_metadata.yml")
    src_dir = os.path.join("temp_extracted", "OS Answer Sheets", "OS_End_Sem_students", "assignment_7244773_export")
    dest_dir = "student_pdfs"

    print(f"Reading metadata from: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        print("Error: Empty metadata loaded.")
        return

    copied_count = 0
    for pdf_file, info in data.items():
        if not pdf_file.endswith(".pdf"):
            continue
        src_file = os.path.join(src_dir, pdf_file)
        if not os.path.exists(src_file):
            print(f"Warning: Source PDF file {src_file} does not exist.")
            continue
        
        submitters = info.get(":submitters", [])
        if submitters:
            # Match the first submitter's roll number (SID)
            sid = submitters[0].get(":sid")
            if sid:
                # Sanitize roll number for filenames
                sid_clean = "".join(c for c in sid if c.isalnum() or c in ("-", "_")).strip()
                dest_file = os.path.join(dest_dir, f"{sid_clean}.pdf")
                shutil.copy2(src_file, dest_file)
                copied_count += 1
                print(f"[{copied_count}] Copied {pdf_file} -> {sid_clean}.pdf")
            else:
                print(f"Warning: No SID found for submission {pdf_file}")
        else:
            print(f"Warning: No submitters info found for submission {pdf_file}")

    print(f"\nSuccessfully prepared {copied_count} student papers in '{dest_dir}' directory.")

if __name__ == "__main__":
    main()
