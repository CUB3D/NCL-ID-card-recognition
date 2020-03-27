#!/usr/bin/env python3
from src import detect
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage ./detect-cli.py <image>")
    exit(1)

if __name__ == "__main__":
    data = detect.extract_card_info(cv2.imread(sys.argv[1]))
    print(f"Status: {data['Status']}")
    print(f"StudentID: {data['StudentID']}")
    print(f"LibraryID: {data['LibraryID']}")
