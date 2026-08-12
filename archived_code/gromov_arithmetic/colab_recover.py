"""Re-adopt or terminate Colab runtimes the CLI has lost the name of.

The CLI addresses runtimes only by a *local* name kept in
``~/.config/colab-cli/sessions.json``.  Lose that mapping and a live VM becomes
unreachable through every normal command while it goes on holding the GPU quota --
``colab sessions`` prints it as ``[?] <endpoint>`` and nothing else will touch it.

It is lost more easily than it looks.  ``StateStore._load_raw`` catches every exception
and returns ``{}``, so a read that lands mid-write yields "no sessions", and the next
``add``/``remove`` persists that empty dict over the real one.  Two pollers hitting the
file at once is enough; it happened twice during this campaign, and the second time the
file was truncated to ``{}`` with a T4 still running.

The server side has no such problem: ``list_assignments`` returns the endpoint together
with the proxy URL and token, which is everything ``SessionState`` needs. So an orphan
can be adopted back under a name and then used or stopped normally.

    python colab_recover.py --list
    python colab_recover.py --adopt gpu-t4-s-kkb-... --name grank
    python colab_recover.py --stop  gpu-t4-s-kkb-...
    python colab_recover.py --stop-all          # frees every assignment

``--stop`` is the one to reach for when a campaign is finished and the quota should go
back; ``--adopt`` first if there are results still on the VM.
"""

from __future__ import annotations

import argparse
import sys

try:
    from colab_cli.auth import get_credentials
    from colab_cli.client import Client, Prod
    from colab_cli.common import State
    from colab_cli.state import SessionState
except ImportError:  # pragma: no cover - depends on the CLI's own environment
    raise SystemExit(
        "run this with the CLI's interpreter, which has colab_cli importable:\n"
        "  & \"$env:APPDATA\\uv\\tools\\google-colab-cli\\Scripts\\python.exe\" "
        "colab_recover.py --list")


def _value(x):
    """The string form of an enum member.

    ``SessionState`` declares ``variant`` and ``accelerator`` as ``str``, but the
    assignment listing returns enums whose ``.value`` is an int for ``variant`` (1 for
    GPU) and a string for ``accelerator`` ("T4"). Taking ``.value`` blindly therefore
    fails validation on one field and works on the other; prefer the name whenever the
    value is not already a string.
    """
    v = getattr(x, "value", x)
    if isinstance(v, str):
        return v
    return getattr(x, "name", str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show server-side assignments")
    ap.add_argument("--adopt", metavar="ENDPOINT", help="re-register under a local name")
    ap.add_argument("--name", default="recovered", help="local name for --adopt")
    ap.add_argument("--stop", metavar="ENDPOINT", help="unassign one runtime")
    ap.add_argument("--stop-all", action="store_true", help="unassign every runtime")
    args = ap.parse_args()

    state = State()
    creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
    client = Client(Prod(), creds)
    assignments = client.list_assignments()

    if not assignments:
        print("no server-side assignments -- nothing is running")
        return 0

    print("assigned runtimes:")
    for a in assignments:
        print(f"  {a.endpoint}  {_value(a.accelerator)}  {_value(a.variant)}")
    known = {a.endpoint: a for a in assignments}

    if args.adopt:
        a = known.get(args.adopt)
        if a is None:
            raise SystemExit(f"\n{args.adopt} is not assigned")
        state.store.add(SessionState(
            name=args.name, token=a.runtime_proxy_info.token,
            url=a.runtime_proxy_info.url, endpoint=a.endpoint,
            variant=_value(a.variant), accelerator=_value(a.accelerator)))
        print(f"\nadopted {a.endpoint} as '{args.name}'. "
              f"`colab exec -s {args.name} ...` should work now.")
        print("Note the keep-alive daemon is NOT restored: the VM may be reclaimed if "
              "left idle, so fetch results promptly.")

    for endpoint in ([a.endpoint for a in assignments] if args.stop_all
                     else ([args.stop] if args.stop else [])):
        if endpoint not in known:
            print(f"\n{endpoint} is not assigned, skipping")
            continue
        print(f"\nunassigning {endpoint} ...")
        client.unassign(endpoint)
        print("  done")

    if args.stop or args.stop_all:
        left = client.list_assignments()
        print("\nstill assigned: " +
              (", ".join(a.endpoint for a in left) if left else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
