"""v0.11.0: 11 hidden short aliases for long commands.

Source change (volcano/cli.py, NOT touched here): each command function got one
extra ``@app.command("<alias>", hidden=True)`` decorator. Mapping:

    who → whoami            ga  → grant-admin (grant_admin_cmd)
    reg → register          sq  → set-queue  (set_queue_cmd)
    img → images            st  → status
    log → logs              fwd → forward
    q   → queue             use → usage
    vol → volumes           (volumes already had vols)

Full names and pre-existing aliases (ls / vols / submit) are unchanged.

Everything runs through Typer's CliRunner with ``sdk.*`` mocked — no kubeconfig,
no cluster, no network, no real ``time.sleep``. Only tests/ is touched.
"""

import re

import pytest
from typer.testing import CliRunner

import volcano.cli as cli

runner = CliRunner()

# alias -> (full command name, underlying callback __name__)
ALIAS_MAP = {
    "who": ("whoami", "whoami"),
    "ga": ("grant-admin", "grant_admin_cmd"),
    "reg": ("register", "register"),
    "sq": ("set-queue", "set_queue_cmd"),
    "img": ("images", "images"),
    "st": ("status", "status"),
    "log": ("logs", "logs"),
    "fwd": ("forward", "forward"),
    "q": ("queue", "queue"),
    "use": ("usage", "usage"),
    "vol": ("volumes", "volumes"),
}

# Pre-existing full command names + old aliases that must stay intact.
PREEXISTING_NAMES = {
    "login", "whoami", "grant-admin", "register", "set", "set-queue", "train",
    "submit", "dev", "save", "images", "list", "ls", "status", "logs", "exec",
    "ssh", "forward", "kill", "volumes", "vols", "queue", "top", "usage",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _guard_no_kube(monkeypatch):
    """A real cluster client must never be built in these unit tests."""
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )
    # current_namespace hits kube for some commands (volumes); stub it out.
    monkeypatch.setattr(cli, "current_namespace", lambda team=None: team or "alice")


def _registry():
    """effective-command-name -> typer CommandInfo for app.registered_commands."""
    out = {}
    for c in cli.app.registered_commands:
        name = c.name or (c.callback.__name__ if c.callback else "?")
        out.setdefault(name, []).append(c)
    return out


def _visible_command_names(help_output: str) -> set:
    """Command-name column of ``volcano --help``.

    Typer/rich renders each command row as ``│ <name>  <desc>`` — exactly one
    space after the box char, then the name. Description continuation lines are
    ``│`` followed by many spaces (the name column blank), so a single-space
    anchor never captures them. Loose by design: we only need the set of names
    that are actually *listed*.
    """
    names = set()
    for line in help_output.splitlines():
        m = re.match(r"^\s*[│|] (\S+)", line)
        if m and m.group(1) != "--help":
            names.add(m.group(1))
    return names


# =========================================================================== #
# 1. registration: every alias exists, is hidden, and points at the same
#    callback as its full name; the full name still exists and is NOT hidden.
# =========================================================================== #
@pytest.mark.parametrize("alias,full,cb", [(a, f, c) for a, (f, c) in ALIAS_MAP.items()])
def test_alias_registered_hidden_same_callback(alias, full, cb):
    reg = _registry()
    assert alias in reg, f"alias {alias!r} not registered"
    assert full in reg, f"full name {full!r} not registered"
    # alias registered exactly once (no accidental double / collision)
    assert len(reg[alias]) == 1, f"{alias!r} registered {len(reg[alias])} times"

    alias_ci = reg[alias][0]
    full_ci = reg[full][0]
    # same underlying function
    assert alias_ci.callback.__name__ == cb
    assert full_ci.callback.__name__ == cb
    assert alias_ci.callback is full_ci.callback
    # alias hidden, full name visible
    assert alias_ci.hidden is True, f"{alias!r} should be hidden"
    assert not full_ci.hidden, f"{full!r} should stay visible"


def test_all_eleven_aliases_present():
    reg = _registry()
    missing = [a for a in ALIAS_MAP if a not in reg]
    assert not missing, f"missing aliases: {missing}"
    assert len(ALIAS_MAP) == 11


# =========================================================================== #
# 2. every alias AND full name responds to --help with exit 0 (no "No such
#    command"). --help short-circuits before the callback, so no sdk needed.
# =========================================================================== #
@pytest.mark.parametrize("name", sorted(ALIAS_MAP) + sorted({f for f, _ in ALIAS_MAP.values()}))
def test_help_runs_exit_zero(name):
    result = runner.invoke(cli.app, [name, "--help"])
    assert result.exit_code == 0, f"{name} --help exit={result.exit_code}\n{result.output}"
    assert "No such command" not in result.output
    assert "Error" not in result.output


def test_alias_and_full_help_describe_same_command():
    # The two --help outputs share the command's docstring / usage body; the only
    # systematic difference is the invoked name echoed in the Usage line.
    for alias, (full, _cb) in ALIAS_MAP.items():
        ra = runner.invoke(cli.app, [alias, "--help"])
        rf = runner.invoke(cli.app, [full, "--help"])
        assert ra.exit_code == 0 and rf.exit_code == 0
        # strip the Usage line (which carries the differing command token)
        strip = lambda o: "\n".join(l for l in o.splitlines() if "Usage:" not in l)
        assert strip(ra.output) == strip(rf.output), f"{alias} vs {full} help differ"


# =========================================================================== #
# 3. no conflict: the 11 alias names are all NEW (none shadows a pre-existing
#    command / old alias); full names still resolve.
# =========================================================================== #
def test_aliases_are_new_names_no_shadowing():
    for alias in ALIAS_MAP:
        assert alias not in PREEXISTING_NAMES, f"{alias!r} collides with an existing name"


def test_full_names_still_invokable(monkeypatch):
    _guard_no_kube(monkeypatch)
    # A representative full name still works end-to-end (whoami has no args).
    monkeypatch.setattr(cli.sdk, "whoami", lambda: {
        "namespace": "alice", "is_admin": False, "team": None, "default_queue": "default",
    })
    result = runner.invoke(cli.app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "namespace" in result.output


# =========================================================================== #
# 4. hidden: none of the 11 aliases appears as a listed command in the
#    top-level ``volcano --help``; the full names DO appear (positive control).
# =========================================================================== #
def test_aliases_hidden_from_top_level_help():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    visible = _visible_command_names(result.output)
    # sanity: parser actually found the command table
    assert "login" in visible and "train" in visible
    for alias, (full, _cb) in ALIAS_MAP.items():
        assert alias not in visible, f"hidden alias {alias!r} leaked into --help"
        assert full in visible, f"full name {full!r} missing from --help"


def test_preexisting_hidden_aliases_still_hidden():
    # ls / vols / submit were already hidden; make sure nothing un-hid them.
    result = runner.invoke(cli.app, ["--help"])
    visible = _visible_command_names(result.output)
    for old in ("ls", "vols", "submit"):
        assert old not in visible, f"{old!r} should remain hidden"


# =========================================================================== #
# 5. behavior equivalence — the three explicitly requested pairs.
# =========================================================================== #
def test_who_equiv_whoami(monkeypatch):
    _guard_no_kube(monkeypatch)
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        return {"namespace": "alice", "is_admin": True,
                "team": "redteam", "default_queue": "redteam"}

    monkeypatch.setattr(cli.sdk, "whoami", fake)
    r_full = runner.invoke(cli.app, ["whoami"])
    r_alias = runner.invoke(cli.app, ["who"])
    assert r_full.exit_code == 0 and r_alias.exit_code == 0, r_alias.output
    assert calls["n"] == 2  # both invocations hit the same sdk.whoami
    assert r_full.output == r_alias.output
    assert "管理员" in r_alias.output


def test_reg_equiv_register_dry_run(monkeypatch):
    _guard_no_kube(monkeypatch)
    seen = []

    def fake_register(team, **kw):
        seen.append((team, kw.get("dry_run")))
        return {"plan": ["namespace redteam", "pvc redteam-nas"]}

    monkeypatch.setattr(cli.sdk, "register_team", fake_register)
    # guard: dry-run must never write a kubeconfig file
    r_full = runner.invoke(cli.app, ["register", "redteam", "--dry-run"])
    r_alias = runner.invoke(cli.app, ["reg", "redteam", "--dry-run"])
    assert r_full.exit_code == 0 and r_alias.exit_code == 0, r_alias.output
    assert seen == [("redteam", True), ("redteam", True)]
    assert r_full.output == r_alias.output
    assert "[dry-run]" in r_alias.output
    assert "redteam-nas" in r_alias.output


def test_q_equiv_queue(monkeypatch):
    _guard_no_kube(monkeypatch)
    calls = {"n": 0}

    def fake_list_queues():
        calls["n"] += 1
        return [{
            "name": "redteam", "state": "Open",
            "allocated": {"gpu": 2, "cpu_m": 16000, "mem_gi": 64},
            "deserved": {"gpu": 4, "cpu_m": 0, "mem_gi": 0},
            "capability": {"gpu": 8, "cpu_m": 0, "mem_gi": 0},
        }]

    monkeypatch.setattr(cli.sdk, "list_queues", fake_list_queues)
    r_full = runner.invoke(cli.app, ["queue"])
    r_alias = runner.invoke(cli.app, ["q"])
    assert r_full.exit_code == 0 and r_alias.exit_code == 0, r_alias.output
    assert calls["n"] == 2
    assert r_full.output == r_alias.output
    assert "redteam" in r_alias.output


# =========================================================================== #
# 5b. behavior equivalence — the remaining read-only / dry-run-safe pairs,
#     data-driven. Each mocks exactly the sdk entry point the command uses,
#     then asserts alias and full name produce identical output and both fire
#     the sdk call. (forward is subprocess/kubectl-bound → help-only, above.)
# =========================================================================== #
_EQUIV_CASES = {
    "ga": {  # grant-admin
        "sdk": "grant_admin",
        "ret": {"sa": "alice", "role": "volcano-admin"},
        "args": ["alice", "--reason", "onboarding"],
        "expect": "管理员",
    },
    "sq": {  # set-queue, view mode (no change flags)
        "sdk": "set_queue",
        "ret": {"name": "redteam",
                "queue": {"deserved": "4", "capability": "8",
                          "reclaimable": True, "state": "Open"}},
        "args": ["redteam"],
        "expect": "deserved=4",
    },
    "img": {  # images
        "sdk": "list_images",
        "ret": [{"kind": "base", "image": "reg/base:1", "team": "", "created": ""}],
        "args": [],
        "expect": "reg/base:1",
    },
    "st": {  # status
        "sdk": "status",
        "ret": {"name": "j", "phase": "Running", "gpu": 8, "cpu_m": 16000,
                "mem_gi": 64,
                "pods": [{"name": "j-worker-0", "phase": "Running", "node": "wuji-0"}]},
        "args": ["j"],
        "expect": "Running",
    },
    "log": {  # logs (non-follow)
        "sdk": "logs",
        "ret": "hello from the pod",
        "args": ["j"],
        "expect": "hello from the pod",
    },
    "use": {  # usage
        "sdk": "usage_by_namespace",
        "ret": [{"namespace": "alice", "gpu": 8, "cpu_m": 16000,
                 "mem_gi": 64, "pods": 2}],
        "args": [],
        "expect": "alice",
    },
    "vol": {  # volumes
        "sdk": "list_pvcs",
        "ret": [{"name": "alice-nas", "status": "Bound", "capacity": "10Ti",
                 "storageclass": "nfs", "volume": "pv1"}],
        "args": [],
        "expect": "alice-nas",
    },
}


@pytest.mark.parametrize("alias", sorted(_EQUIV_CASES))
def test_alias_equiv_full_behavior(alias, monkeypatch):
    _guard_no_kube(monkeypatch)
    full = ALIAS_MAP[alias][0]
    case = _EQUIV_CASES[alias]
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return case["ret"]

    monkeypatch.setattr(cli.sdk, case["sdk"], fake)
    r_full = runner.invoke(cli.app, [full] + case["args"])
    r_alias = runner.invoke(cli.app, [alias] + case["args"])
    assert r_full.exit_code == 0, f"{full}: {r_full.output}"
    assert r_alias.exit_code == 0, f"{alias}: {r_alias.output}"
    assert calls["n"] == 2, f"{alias}/{full} should both hit sdk.{case['sdk']}"
    assert r_full.output == r_alias.output
    assert case["expect"] in r_alias.output


# =========================================================================== #
# 6. adversarial: alias inherits identical option surface (a bad flag fails the
#    same way on alias and full name), and the alias forwards args, not eats them.
# =========================================================================== #
def test_alias_rejects_unknown_option_like_full():
    r_full = runner.invoke(cli.app, ["status", "--nope"])
    r_alias = runner.invoke(cli.app, ["st", "--nope"])
    assert r_full.exit_code != 0 and r_alias.exit_code != 0
    # both should be a usage error, not a crash/traceback
    assert r_full.exception is None or isinstance(r_full.exception, SystemExit)
    assert r_alias.exception is None or isinstance(r_alias.exception, SystemExit)


def test_alias_forwards_positional_and_team(monkeypatch):
    # `st <name> -t <team>` must thread name+team into sdk.status exactly like status.
    _guard_no_kube(monkeypatch)
    seen = {}

    def fake_status(name, team=None):
        seen["name"], seen["team"] = name, team
        return {"name": name, "phase": "Pending", "gpu": 0, "cpu_m": 0,
                "mem_gi": 0, "pods": []}

    monkeypatch.setattr(cli.sdk, "status", fake_status)
    result = runner.invoke(cli.app, ["st", "myjob", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "myjob", "team": "teamx"}
