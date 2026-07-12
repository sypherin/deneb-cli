#!/bin/sh
# Deneb installer — the open MIT client for Altronis Neo local-LLM setup.
#
#   curl -fsSL https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh
#
# Installs `deneb` via pipx (isolated, no system Python pollution). Safe to re-run to
# upgrade. Does NOT need a GitHub account, and never asks for sudo.
set -eu

REPO="git+https://github.com/sypherin/deneb-cli.git"
say() { printf '\033[38;5;44m▸\033[0m %s\n' "$1"; }
err() { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

command -v python3 >/dev/null 2>&1 || {
  err "python3 not found — install Python 3.9+ first, then re-run this."; exit 1; }
command -v git >/dev/null 2>&1 || {
  err "git not found — install git first (needed to fetch the client), then re-run."; exit 1; }

if ! command -v pipx >/dev/null 2>&1 && ! python3 -m pipx --version >/dev/null 2>&1; then
  say "installing pipx (one-time)…"
  python3 -m pip install --user -q pipx || {
    err "couldn't install pipx. Try:  python3 -m pip install --user pipx"; exit 1; }
  python3 -m pipx ensurepath >/dev/null 2>&1 || true
fi

say "installing deneb…"
if command -v pipx >/dev/null 2>&1; then
  pipx install --force "$REPO"
else
  python3 -m pipx install --force "$REPO"
fi

echo
say "installed. next:"
echo "    deneb auth --token <your-token>     # sign in (paste once)"
echo "    deneb check                         # scan this box — am I done?"
echo "    deneb                               # interactive troubleshooting"
echo
printf '\033[2mif "deneb" is not found, open a new terminal — pipx just added ~/.local/bin to your PATH.\033[0m\n'
