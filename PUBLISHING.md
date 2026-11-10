# Publishing this repository (GitHub + Zenodo)

## Quick way: one command

Install the GitHub CLI (https://cli.github.com) if you do not have it, open a terminal in this folder, and run:

- **Windows (PowerShell):** `powershell -ExecutionPolicy Bypass -File publish_cmfs.ps1`
- **macOS, Linux or Git Bash:** `bash publish_cmfs.sh`

The script signs you in to GitHub (in your browser, the first time only), creates the public repository `cmfs`, fills in its address in the README and `CITATION.cff`, pushes the code and the `v0.2.0` tag, pauses while you switch the repository on in Zenodo, creates the release that triggers the DOI, and records the DOI if you paste it. Afterwards, replace `USERNAME` and `XXXXXXX` in the manuscript as the script tells you.

## Manual way

These steps are for the author; they take about 15 minutes.

## 1. Create the GitHub repository
1. Sign in to GitHub and click **New repository**. Name it `cmfs`, set it to **Public**, and do **not** add a README, licence or .gitignore (they are already here).
2. Upload the code, either:
   - **Command line** (this folder is already a git repository with the v0.2.0 commit and tag):

         git remote add origin https://github.com/USERNAME/cmfs.git
         git push -u origin main
         git push origin v0.2.0

   - **Or the website:** open the new repository, click **uploading an existing file**, drag in the contents of this folder (not the folder itself), and commit.

## 2. Connect Zenodo (once)
1. Sign in to https://zenodo.org with your GitHub account.
2. Open **Settings > GitHub**, find `cmfs`, and switch it **On**.

## 3. Make the release (this creates the DOI)
1. On GitHub, open **Releases > Draft a new release**.
2. Choose the tag `v0.2.0` (create it on the website if you uploaded through the browser), title it `CMFS v0.2.0`, and publish.
3. Within a few minutes Zenodo archives the release and shows its DOI (10.5281/zenodo.XXXXXXX).

## 4. Fill in the placeholders
Replace `USERNAME` and `XXXXXXX` in:
- `README.md` (citation section and the DOI badge comment at the top),
- `CITATION.cff` (`repository-code` and the `doi` line),
- the manuscript: the *Software and Data Availability* section and the software entry in `cas-refs.bib`.

Commit these changes. For later code changes (for example after peer review), release `v0.2.1`; Zenodo creates a new DOI for each version, and the paper should cite the version that matches its final results.
