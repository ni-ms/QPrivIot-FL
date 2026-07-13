<#
.SYNOPSIS
    Build paper/qpriviot-memory.pdf.

.DESCRIPTION
    Runs the full pdflatex -> bibtex -> pdflatex -> pdflatex cycle and reports
    errors, undefined references, overfull boxes and bibtex warnings. Exits
    non-zero if the build fails, so it is safe to chain or run in CI.

    MiKTeX is installed user-scope and is NOT on the global PATH, so this script
    prepends it. Missing LaTeX packages are auto-installed by MiKTeX on first use.

.PARAMETER Figures
    Regenerate figures/fig_*.pdf from experiment_results/ before building. Only
    needed if the experiment grid has been re-run; the committed figures are
    already current.

.PARAMETER Clean
    Delete build artifacts (.aux/.bbl/.blg/.log/.out) and the PDF, then exit.

.PARAMETER Quick
    Single pdflatex pass, no bibtex. Fast prose-only check; cross-references and
    citations will be stale. Do not use for a final PDF.

.EXAMPLE
    .\build.ps1
.EXAMPLE
    .\build.ps1 -Figures
.EXAMPLE
    .\build.ps1 -Clean
#>
[CmdletBinding()]
param(
    [switch]$Figures,
    [switch]$Clean,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$JOB  = 'qpriviot-memory'
$ROOT = Split-Path $PSScriptRoot -Parent
$ARTIFACTS = @("$JOB.aux", "$JOB.bbl", "$JOB.blg", "$JOB.log", "$JOB.out",
               "$JOB.synctex.gz", "$JOB.fls", "$JOB.fdb_latexmk", "$JOB.spl")

if ($Clean) {
    $ARTIFACTS + @("$JOB.pdf") | ForEach-Object {
        if (Test-Path $_) { Remove-Item $_ -Force; Write-Host "  removed $_" }
    }
    Write-Host "clean." -ForegroundColor Green
    exit 0
}

# MiKTeX is user-scope and off the global PATH.
$miktex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$miktex\pdflatex.exe")) {
    Write-Host "pdflatex not found at $miktex" -ForegroundColor Red
    Write-Host "Install it with:  winget install MiKTeX.MiKTeX" -ForegroundColor Yellow
    exit 1
}
$env:PATH = "$miktex;$env:PATH"

# bibtex resolves the .bst from the working directory, not from bst/.
if (-not (Test-Path 'sn-mathphys-num.bst')) {
    Copy-Item 'bst\sn-mathphys-num.bst' . -Force
}

if ($Figures) {
    Write-Host "==> regenerating figures from experiment_results/" -ForegroundColor Cyan
    $py = Join-Path $ROOT '.venv\Scripts\python.exe'
    & $py (Join-Path $ROOT 'scripts\agentmem\rerun_figures.py') --grid rerun_grid_rp
    if ($LASTEXITCODE -ne 0) { Write-Host "figure regeneration failed" -ForegroundColor Red; exit 1 }
}

# Start from a clean slate so a stale .aux cannot mask a broken build.
$ARTIFACTS | ForEach-Object { if (Test-Path $_) { Remove-Item $_ -Force } }

$tmp    = [System.IO.Path]::GetTempPath()
$log    = Join-Path $tmp "$JOB.build.log"     # stdout of the most recent pass
$errLog = Join-Path $tmp "$JOB.build.err"     # stderr of the most recent pass

function Invoke-Pass {
    # Two PowerShell 5.1 landmines are deliberately avoided here:
    #
    #  1. Do NOT name a parameter $Args -- it collides with the automatic $args
    #     variable, the exe then receives no arguments, and pdflatex hangs forever
    #     at its interactive "**" filename prompt.
    #
    #  2. Do NOT redirect a native exe's stderr with `*>` or `2>&1`. PS 5.1 wraps
    #     each stderr line in an ErrorRecord (NativeCommandError) and, under
    #     $ErrorActionPreference='Stop', kills the script -- so MiKTeX's harmless
    #     "you have not checked for updates" nag reads as a fatal build failure.
    #     Start-Process writes the streams to files without that wrapping, and
    #     gives us a trustworthy ExitCode.
    param([string]$Label, [string]$Exe, [string[]]$ExeArgs)

    Write-Host "==> $Label" -ForegroundColor Cyan
    $p = Start-Process -FilePath $Exe -ArgumentList $ExeArgs `
                       -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $log -RedirectStandardError $errLog

    # pdflatex exits non-zero on a real error. bibtex reports problems in its own
    # .blg (checked in the report below) and can exit non-zero on mere warnings,
    # so we do not treat its exit code as fatal.
    if ($p.ExitCode -ne 0 -and $Exe -eq 'pdflatex') {
        Write-Host "`n$Label FAILED (exit $($p.ExitCode)):" -ForegroundColor Red
        Select-String -Path $log -Pattern '^!' -Context 0, 3 |
            Select-Object -First 5 |
            ForEach-Object { Write-Host "  $($_.Line)"; $_.Context.PostContext | ForEach-Object { Write-Host "  $_" } }
        Write-Host "`nfull log: $log" -ForegroundColor Yellow
        exit 1
    }
}

# -interaction=nonstopmode also guarantees pdflatex never blocks on a prompt, so a
# hang here means a bad invocation rather than a bad document.
$pdflatexArgs = @('-interaction=nonstopmode', '-halt-on-error', "$JOB.tex")

if ($Quick) {
    Invoke-Pass 'pdflatex (quick, refs will be stale)' 'pdflatex' $pdflatexArgs
} else {
    Invoke-Pass 'pdflatex (1/3)' 'pdflatex' $pdflatexArgs
    Invoke-Pass 'bibtex'         'bibtex'   @($JOB)
    Invoke-Pass 'pdflatex (2/3)' 'pdflatex' $pdflatexArgs
    Invoke-Pass 'pdflatex (3/3)' 'pdflatex' $pdflatexArgs
}

# --- report ---------------------------------------------------------------
$errors    = @(Select-String -Path $log -Pattern '^!')
$undefined = @(Select-String -Path $log -Pattern 'undefined on input|Citation .* undefined')
$overfull  = @(Select-String -Path $log -Pattern 'Overfull')
$bibwarn   = if (Test-Path "$JOB.blg") { @(Select-String -Path "$JOB.blg" -Pattern 'Warning--') } else { @() }
$entries   = if (Test-Path "$JOB.bbl") { @(Select-String -Path "$JOB.bbl" -Pattern '^.bibitem') } else { @() }
$pages     = (Select-String -Path $log -Pattern 'Output written .*\((\d+) pages' |
              Select-Object -Last 1).Matches.Groups[1].Value

function Show-Count {
    param([string]$Name, [int]$N, [switch]$WantZero)
    $ok = if ($WantZero) { $N -eq 0 } else { $N -gt 0 }
    $c  = if ($ok) { 'Green' } else { 'Yellow' }
    Write-Host ("  {0,-22} {1}" -f "$Name`:", $N) -ForegroundColor $c
}

Write-Host "`n--- build report ---" -ForegroundColor Cyan
Show-Count 'errors'          $errors.Count    -WantZero
Show-Count 'undefined refs'  $undefined.Count -WantZero
Show-Count 'overfull boxes'  $overfull.Count  -WantZero
Show-Count 'bibtex warnings' $bibwarn.Count   -WantZero
if (-not $Quick) { Show-Count 'bib entries' $entries.Count }

if ($undefined.Count -gt 0) {
    Write-Host "`n  undefined (first 5):" -ForegroundColor Yellow
    $undefined | Select-Object -First 5 | ForEach-Object { "    $($_.Line.Trim())" }
}
if ($overfull.Count -gt 0) {
    Write-Host "`n  overfull (first 5):" -ForegroundColor Yellow
    $overfull | Select-Object -First 5 | ForEach-Object { "    $($_.Line.Trim())" }
}

if (Test-Path "$JOB.pdf") {
    $kb = [math]::Round((Get-Item "$JOB.pdf").Length / 1KB)
    Write-Host "`nwrote $JOB.pdf  ($pages pages, $kb KB)" -ForegroundColor Green
    Write-Host "full log: $log"
    exit 0
} else {
    Write-Host "`nno PDF produced. log: $log" -ForegroundColor Red
    exit 1
}
