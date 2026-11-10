# Publish CMFS to GitHub and trigger the Zenodo DOI (Windows PowerShell).
# Requirements: git and the GitHub CLI "gh" (https://cli.github.com). Usage:  powershell -ExecutionPolicy Bypass -File publish_cmfs.ps1 [-RepoName cmfs]
param([string]$RepoName = "cmfs")
$ErrorActionPreference = "Continue"   # git writes progress to stderr; exit codes are checked instead
Set-Location $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Edit-File($path, $from, $to) {
  $full = (Resolve-Path $path).Path
  $text = [IO.File]::ReadAllText($full, $utf8)
  [IO.File]::WriteAllText($full, $text.Replace($from, $to), $utf8)
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Write-Host "git is not installed."; exit 1 }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Host "GitHub CLI (gh) not found. Install it from https://cli.github.com and run this script again,"
  Write-Host "or follow the manual steps in PUBLISHING.md."; exit 1
}
gh auth status 2>$null; if ($LASTEXITCODE -ne 0) { gh auth login }

$owner = (gh api user --jq .login).Trim()
$url = "https://github.com/$owner/$RepoName"
Write-Host "Publishing to $url"

# 1. Fill in the repository address, then (re)tag the release commit
Edit-File "README.md" "github.com/USERNAME/cmfs" "github.com/$owner/$RepoName"
Edit-File "CITATION.cff" "github.com/USERNAME/cmfs" "github.com/$owner/$RepoName"
git diff --quiet; if ($LASTEXITCODE -ne 0) { git commit -qam "Add repository address" }
git tag -f -a v0.2.0 -m "CMFS v0.2.0" | Out-Null

# 2. Create the public repository and push the code and the tag
git remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "Remote 'origin' already set." }
else {
  gh repo create $RepoName --public --source . --remote origin --description "CMFS: Causal Mediation Feature Selection - code for hemisphere-specific stroke biomarker discovery"
  if ($LASTEXITCODE -ne 0) { Write-Host "Could not create the repository."; exit 1 }
}
git push -u origin main;      if ($LASTEXITCODE -ne 0) { Write-Host "Push failed."; exit 1 }
git push -f origin v0.2.0;     if ($LASTEXITCODE -ne 0) { Write-Host "Tag push failed."; exit 1 }

# 3. Zenodo must be switched on BEFORE the release, so that the release is archived
Write-Host ""
Write-Host "Now open https://zenodo.org/account/settings/github/ (sign in with GitHub),"
Write-Host "click 'Sync now' if needed, and switch ON the repository '$owner/$RepoName'."
Read-Host "Press Enter once it is switched on" | Out-Null

# 4. Create the GitHub release; Zenodo archives it and issues the DOI within a few minutes
gh release create v0.2.0 --title "CMFS v0.2.0" --notes "Code accompanying the manuscript 'CMFS: Causal Mediation Biomarker Discovery Based on Tiered Clinical Selection for Hemisphere-Specific Stroke Classification'. Reproduces every result in the paper, including the simulation study."
if ($LASTEXITCODE -ne 0) { Write-Host "Release failed."; exit 1 }
Write-Host "Release created: $url/releases/tag/v0.2.0"

# 5. Optionally record the DOI in the README and CITATION.cff
Write-Host ""
Write-Host "The DOI appears on https://zenodo.org/account/settings/github/ within a few minutes."
$doi = Read-Host "Paste the DOI (e.g. 10.5281/zenodo.1234567), or press Enter to do this later"
if ($doi) {
  $id = $doi.Split(".")[-1]
  Edit-File "README.md" "10.5281/zenodo.XXXXXXX" $doi
  Edit-File "README.md" "zenodo.XXXXXXX" "zenodo.$id"
  $readme = [IO.File]::ReadAllText((Resolve-Path "README.md").Path, $utf8)
  $readme = [regex]::Replace($readme, "<!-- After the first Zenodo release, add the DOI badge here: (.*) -->", '$1')
  [IO.File]::WriteAllText((Resolve-Path "README.md").Path, $readme, $utf8)
  Edit-File "CITATION.cff" "# doi: 10.5281/zenodo.XXXXXXX" "doi: $doi"
  git commit -qam "Add Zenodo DOI $doi"; git push
  Write-Host "DOI recorded. In the manuscript, replace USERNAME with $owner and XXXXXXX with $id."
} else {
  Write-Host "Later: replace XXXXXXX in README.md and CITATION.cff, and USERNAME/XXXXXXX in the manuscript."
}
Write-Host "Done."
