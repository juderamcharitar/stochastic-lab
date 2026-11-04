# STOCHASTIC LAB public release — run in PowerShell after extracting this ZIP.
# Git for Windows is required. GitHub browser login may be requested by Git Credential Manager.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$repo = 'https://github.com/juderamcharitar/stochastic-lab.git'
function GitRun([string[]]$gitArgs) {
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "Git command failed (exit $LASTEXITCODE): git $($gitArgs -join ' ')" }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git for Windows is not installed. Install it from https://git-scm.com/downloads/win then rerun this script.'
}
foreach($path in @('README.md','app.py','OPEN_THIS_NEON_TERRAIN.html','paper/STOCHASTIC_LAB_V115_REVISED_MANUSCRIPT.pdf','src/robustness_v09.py')) {
    if (-not (Test-Path $path)) { throw "Required release file missing: $path" }
}
if (-not (Test-Path '.git')) { GitRun @('init','-b','main') }
$existing = (& git remote get-url origin 2>$null)
if ($LASTEXITCODE -eq 0) {
    if ($existing -ne $repo) { throw "Existing origin points to $existing, not $repo" }
} else {
    GitRun @('remote','add','origin',$repo)
}
GitRun @('add','--all')
# Do not include raw market archives; check the release before committing.
$staged = & git diff --cached --name-only
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect staged files' }
$blocked = @($staged | Where-Object { $_ -match '(^|/)(\.env($|\.)|data/monthly_5m/|.*\.(zip|gz|key|pem)$)' })
if ($blocked.Count -gt 0) { throw "Blocked files detected: $($blocked -join ', ')" }
$status = & git status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Could not check status' }
if ($status) {
    GitRun @('commit','-m','Publish STOCHASTIC LAB V1.2 research release')
} else { Write-Host 'No new files to commit.' }
Write-Host 'Pushing to public repository...'
GitRun @('push','-u','origin','main')
Write-Host 'PUBLISHED: https://github.com/juderamcharitar/stochastic-lab'
