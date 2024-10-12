from pathlib import Path
import subprocess

def generate():
    subprocess.run([
        "wit-bindgen",
        "moonbit",
        "wit",
        "--derive-show",
        "--derive-eq",
        "--derive-error",
        "--out-dir",
        "tmp",
    ])
    Path("tmp/ffi")

def main():
    generate()

if __name__ == "__main__":
    main()
