#!/bin/sh
# Deneb installer: the open MIT tool to check, set up and watch a local LLM server.
#
#   curl -fsSL https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh
#
# Installs `deneb` via pipx (isolated, no system Python pollution). Safe to re-run to
# upgrade. Does NOT need a GitHub account, and never runs sudo. If pipx is missing it tries
# to set it up without sudo; where that is blocked (recent Ubuntu and Debian) it stops and
# prints the command to install pipx.
set -eu

REPO="git+https://github.com/sypherin/deneb-cli.git"
say() { printf '\033[38;5;44m▸\033[0m %s\n' "$1"; }
err() { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

# how <fedora pkg> <debian pkg> <brew pkg>: the install line for each system, then stop.
how() {
  echo "Install it with the command for your system:" >&2
  printf '    %-26s# Fedora\n' "sudo dnf install $1" >&2
  printf '    %-26s# Debian or Ubuntu\n' "sudo apt install $2" >&2
  printf '    %-26s# macOS with Homebrew\n' "brew install $3" >&2
  echo "then re-run this installer." >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || {
  err "python3 not found: Deneb needs Python 3.9 or newer."; how python3 python3 python; }
command -v git >/dev/null 2>&1 || {
  err "git not found: it is needed to fetch the client."; how git git git; }

if ! command -v pipx >/dev/null 2>&1 && ! python3 -m pipx --version >/dev/null 2>&1; then
  say "pipx not found, trying to set it up without sudo (one-time)..."
  if python3 -m pip install --user -q pipx >/dev/null 2>&1; then
    python3 -m pipx ensurepath >/dev/null 2>&1 || true
  else
    err "couldn't set up pipx without sudo (recent Ubuntu and Debian block that)."
    how pipx pipx pipx
  fi
fi

if command -v pipx >/dev/null 2>&1; then PIPX="pipx"; else PIPX="python3 -m pipx"; fi
say "installing deneb…"
$PIPX install --force "$REPO"
# where pipx put the command (older pipx has no `environment --value`: use its default)
BIN=$($PIPX environment --value PIPX_BIN_DIR 2>/dev/null) || BIN=""
[ -n "$BIN" ] || BIN="$HOME/.local/bin"

echo
say "installed. next:"
echo "    deneb check                         # is the model served on the GPU, ready?"
echo "    deneb recommend                     # which models fit this box"
echo "    deneb setup <model>                 # the setup steps for the model you pick"
echo "    deneb watch --schedule              # alert when it breaks or recovers"
echo "    deneb \"why is my model slow\"        # chat, free, with the model on your box"
echo "    deneb --help                        # everything else (all free, no token)"

# A new terminal tab is not enough on Ubuntu and Debian: ~/.profile adds ~/.local/bin to
# PATH only at login, and only if the folder existed then. Say exactly what works.
if ! command -v deneb >/dev/null 2>&1; then
  echo
  printf '\033[33m!\033[0m this terminal cannot find "deneb" yet: it is in %s,\n' "$BIN"
  echo "  which is not on your PATH. Run it with the full path:"
  echo "      $BIN/deneb check"
  echo "  or add the folder to this terminal:"
  printf '      export PATH="%s:$PATH"\n' "$BIN"
  echo "  To fix it for good, log out and back in once. If a new login still cannot"
  echo "  find it, run: pipx ensurepath"
fi
