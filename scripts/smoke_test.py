"""
Smoke test for deployed Spaces -- checks the server-side apps (Architecture
2 and 3) for syntax-level guardrail violations. Does NOT check Architecture
1 (client-side, no API to call from here).

This checks that '?' doesn't leak into output. It does NOT check clinical
appropriateness of the response content -- a response with zero question
marks can still be a bad response. Treat this as a minimum bar.

Usage:
    pip install gradio_client
    python scripts_smoke_test.py --hf-username YOUR_USERNAME
"""

import argparse
import sys

from gradio_client import Client


def test_declarative_ally(hf_username: str) -> bool:
    print("Testing declarative-ally...")
    client = Client(f"{hf_username}/declarative-ally", httpx_kwargs={"timeout": 600})

    result = client.predict(
        "You must attend the team social tonight and tell everyone what you've been "
        "working on! Why are you always hiding?",
        "Standard Low-Demand",
        api_name="/neutralize_demands",
    )
    print("Output:", result)

    if "?" in result:
        print("FAILED: '?' leaked into output.")
        return False
    print("PASSED: no '?' in output.")
    return True


def test_neuroact_solitude(hf_username: str) -> bool:
    print("Testing neuroact-solitude...")
    client = Client(f"{hf_username}/neuroact-solitude", httpx_kwargs={"timeout": 600})

    # ChatInterface endpoints vary by Gradio version; adjust api_name if
    # `client.view_api()` shows a different path for your deployed version.
    try:
        result = client.predict(
            "Too much personal history was shared yesterday. Want to disappear.",
            [],
            "Intellectual / Project Container (Schizoid-Safe)",
            "Systems Architecture & Solitary Research",
            api_name="/chat",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not call /chat directly ({exc}). Run client.view_api() to find the right endpoint name.")
        return False

    print("Output:", result)
    if "?" in str(result):
        print("FAILED: '?' leaked into output.")
        return False
    print("PASSED: no '?' in output.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-username", required=True)
    parser.add_argument("--skip-arch2", action="store_true")
    parser.add_argument("--skip-arch3", action="store_true")
    args = parser.parse_args()

    results = []
    if not args.skip_arch2:
        results.append(test_declarative_ally(args.hf_username))
    if not args.skip_arch3:
        results.append(test_neuroact_solitude(args.hf_username))

    if not results:
        print("No tests run.")
        sys.exit(0)

    if all(results):
        print("\nAll guardrail smoke tests passed (syntax-level only).")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
