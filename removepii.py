# Author: Jose L. Ocana-Pujol (original Porter Zach
# Python 3.12

import argparse
import nltk
import re
import os
import pathlib
from typing import List, Set
exclusions_file = "exclusions.txt"

# Ensure NLTK data is available
def ensure_nltk_data():
    """Ensure all required NLTK data is available."""
    resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for res in resources:
        nltk.download(res, quiet=True)

# Extract text from files
def extract(filePath: str, encoding="utf-8") -> str:
    """Extracts the textual information from a file."""
    ext = pathlib.Path(filePath).suffix

    if ext in [".txt", ".md"]:
        with open(filePath, encoding=encoding) as file:
            return file.read()
    elif ext == ".pdf":
        from pdfminer.high_level import extract_text
        return extract_text(filePath)
    elif ext in [".html", ".htm"]:
        from bs4 import BeautifulSoup
        with open(filePath, encoding=encoding) as file:
            soup = BeautifulSoup(file, "html.parser")
        return soup.get_text()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types are TXT, PDF, HTML.")

def getNE(text: str, piiNE: List[str]) -> Set[str]:
    """Gets the named entities classified as PII in the text."""
    ensure_nltk_data()
    ne = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    pii = set()

        # Read exclusions from the file
    with open(exclusions_file, "r") as file:
        exclusions = {line.strip().lower() for line in file}

    for subtree in ne.subtrees(lambda x: x.label() in piiNE):
        entity = " ".join(word for word, _ in subtree.leaves())
        # Normalize entity and exclusions to lowercase for comparison
        if entity.lower() not in exclusions:
            pii.add(entity)
    return pii

# Get ID info classified as PII
def getIDInfo(text: str, types: List[str]) -> Set[str]:
    """Gets the ID info classified as PII in the text."""
    patterns = {
        "PHONE": re.compile(r'''(
            (\d{3}|\(\d{3}\))?(\s|-|\.)?(\d{3})(\s|-|\.)(\d{4})
            (\s*(ext|x|ext.)\s*(\d{2,5}))?
        )''', re.VERBOSE),
        "EMAIL": re.compile(r'''(
            [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}
        )''', re.VERBOSE),
        "SSN": re.compile(r'''(
            (?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}
        )''', re.VERBOSE)
    }

    pii = set()
    for key, pattern in patterns.items():
        if key in types:
            pii.update(match[0] if isinstance(match, tuple) else match for match in pattern.findall(text))
    return pii

# Write text to a file
def writeFile(text: str, path: str):
    """Writes text to the file path."""
    with open(path, "w") as file:
        file.write(text)

# Clean a string of PII
def cleanString(text: str, verbose=False, piiNE=None, piiNums=None) -> str:
    """Cleans a string of PII."""
    piiNE = piiNE or ["PERSON", "ORGANIZATION", "GPE", "LOCATION"]
    piiNums = piiNums or ["PHONE", "EMAIL", "SSN"]

    if verbose:
        print("Cleaning text: getting named entities and identifiable information...")
    piiSet = getNE(text, piiNE).union(getIDInfo(text, piiNums))
    if verbose:
        print(f"{len(piiSet)} PII strings found.")

    cleaned = text
    for pii in piiSet:
        cleaned = cleaned.replace(pii, "XXXXX")

    cleaned = maskDirectories(cleaned)
    return cleaned

# Clean a file of PII
def cleanFile(filePath: str, outputPath: str, verbose=False, piiNE=None, piiNums=None):
    """Reads a file with PII and saves a copy with PII removed."""
    if verbose:
        print(f"Extracting text from {filePath}...")
    text = extract(filePath)
    if verbose:
        print("Text extracted.")

    cleaned = cleanString(text, verbose, piiNE, piiNums)
    if verbose:
        print(f"Writing clean text to {outputPath}.")
    writeFile(cleaned, outputPath)

# Mask directory paths in text
def maskDirectories(text: str) -> str:
    """Masks directory information in a file path."""
    dir_pattern = re.compile(r'([A-Za-z]:\\[^ \n\r\t]+|/[^ \n\r\t]+)')
    return re.sub(dir_pattern, lambda m: mask_directory(m.group()), text)

def mask_directory(path: str) -> str:
    """Helper function to mask directory paths."""
    path = path.replace('\\', '/')
    parts = path.split('/')
    if len(parts) > 1:
        parts[:-1] = ['XXXXX'] * (len(parts) - 1)
    return '/'.join(parts)

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes personally identifiable information (PII) like names and phone numbers from text strings and files."
    )
    parser.add_argument("-f", nargs=2, dest="path", metavar=("inputPath", "outputPath"),
                        help="The file to remove PII from and the clean output file path.")
    parser.add_argument("-s", dest="text", help="Input a text string to clean.")

    args = parser.parse_args()

    if args.path:
        cleanFile(args.path[0], args.path[1], verbose=True)
    elif args.text:
        print("Text with PII removed:\n" + cleanString(args.text, verbose=True))
    else:
        print("No action specified.")