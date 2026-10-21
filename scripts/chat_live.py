"""
Quick interactive terminal client to talk to your live Hugging Face Spaces.

Usage:
    python scripts/chat_live.py --app arch2
    python scripts/chat_live.py --app arch3
"""

import argparse
import sys
from gradio_client import Client


def chat_declarative(user: str):
    space_id = f"{user}/declarative-ally"
    print(f"\nConnecting to live Space: https://huggingface.co/spaces/{space_id} ...")
    client = Client(space_id)
    print("Connected! Paste high-demand text to neutralize (or Ctrl+C to quit):\n")

    while True:
        try:
            prompt = input("demand> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not prompt:
            continue
        try:
            out = client.predict(prompt, "Standard Low-Demand", api_name="/neutralize_demands")
            print(f"\n[Declarative-Ally]\n{out}\n")
        except Exception as e:
            print(f"[Error]: {e}\n")


def chat_companion(user: str):
    space_id = f"{user}/neuroact-solitude"
    print(f"\nConnecting to live Space: https://huggingface.co/spaces/{space_id} ...")
    client = Client(space_id)
    print("Connected! Type a message (or Ctrl+C to quit):\n")

    while True:
        try:
            msg = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not msg:
            continue
        try:
            out = client.predict(
                msg,
                "Intellectual / Project Container (Schizoid-Safe)",
                "Systems Architecture & Solitary Research",
                api_name="/chat",
            )
            print(f"\n[NeuroACT-Solitude]\n{out}\n")
        except Exception as e:
            print(f"[Error]: {e}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-username", default="Scadiva")
    p.add_argument("--app", choices=["arch2", "arch3"], default="arch3", help="Which Space to test (arch2 or arch3)")
    args = p.parse_args()

    if args.app == "arch2":
        chat_declarative(args.hf_username)
    else:
        chat_companion(args.hf_username)


if __name__ == "__main__":
    main()
