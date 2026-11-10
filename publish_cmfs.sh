#!/usr/bin/env bash
# Publish CMFS to GitHub and trigger the Zenodo DOI (macOS, Linux, or Git Bash on Windows).
# Requirements: git and the GitHub CLI "gh" (https://cli.github.com). Usage:  bash publish_cmfs.sh [repository-name]
set -euo pipefail
REPO_NAME="${1:-cmfs}"
cd "$(dirname "$0")"

command -v git >/dev/null || { echo "git is not installed."; exit 1; }
if ! command -v gh >/dev/null; then
  echo "GitHub CLI (gh) not found. Install it from https://cli.github.com and run this script again,"
  echo "or follow the manual steps in PUBLISHING.md."; exit 1
fi
gh auth status >/dev/null 2>&1 || gh auth login

OWNER=$(gh api user --jq .login)
URL="https://github.com/$OWNER/$REPO_NAME"
echo "Publishing to $URL"

# 1. Fill in the repository address, then (re)tag the release commit
sed -i.bak "s#github.com/USERNAME/cmfs#github.com/$OWNER/$REPO_NAME#g" README.md CITATION.cff && rm -f README.md.bak CITATION.cff.bak
if ! git diff --quiet; then git commit -qam "Add repository address"; fi
git tag -f -a v0.2.0 -m "CMFS v0.2.0" >/dev/null

# 2. Create the public repository and push the code and the tag
if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' already set: $(git remote get-url origin)"
else
  gh repo create "$REPO_NAME" --public --source . --remote origin \
     --description "CMFS: Causal Mediation Feature Selection - code for hemisphere-specific stroke biomarker discovery"
fi
git push -u origin main
git push -f origin v0.2.0

# 3. Zenodo must be switched on BEFORE the release, so that the release is archived
echo
echo "Now open https://zenodo.org/account/settings/github/ (sign in with GitHub),"
echo "click 'Sync now' if needed, and switch ON the repository '$OWNER/$REPO_NAME'."
read -r -p "Press Enter once it is switched on... " _

# 4. Create the GitHub release; Zenodo archives it and issues the DOI within a few minutes
gh release create v0.2.0 --title "CMFS v0.2.0" \
   --notes "Code accompanying the manuscript 'CMFS: Causal Mediation Biomarker Discovery Based on Tiered Clinical Selection for Hemisphere-Specific Stroke Classification'. Reproduces every result in the paper, including the simulation study."
echo "Release created: $URL/releases/tag/v0.2.0"

# 5. Optionally record the DOI in the README and CITATION.cff
echo
echo "The DOI appears on https://zenodo.org/account/settings/github/ within a few minutes."
read -r -p "Paste the DOI (e.g. 10.5281/zenodo.1234567), or press Enter to do this later: " DOI
if [ -n "$DOI" ]; then
  ID="${DOI##*.}"
  sed -i.bak "s#10.5281/zenodo.XXXXXXX#$DOI#g; s#zenodo.XXXXXXX#zenodo.$ID#g; s#<!-- After the first Zenodo release, add the DOI badge here: \(.*\) -->#\1#" README.md && rm -f README.md.bak
  sed -i.bak "s|^# doi: 10.5281/zenodo.XXXXXXX|doi: $DOI|" CITATION.cff && rm -f CITATION.cff.bak
  git commit -qam "Add Zenodo DOI $DOI" && git push
  echo "DOI recorded. In the manuscript, replace USERNAME with $OWNER and XXXXXXX with $ID."
else
  echo "Later: replace XXXXXXX in README.md and CITATION.cff, and USERNAME/XXXXXXX in the manuscript."
fi
echo "Done."
