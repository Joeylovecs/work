$ErrorActionPreference = 'Stop'

$projectPath = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$repoPath = $projectPath
$pollSeconds = 15
$debounceSeconds = 30
$retrySeconds = 60

Set-Location -LiteralPath $repoPath

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    & git -C $repoPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-GitStatus {
    @( & git -C $repoPath status --porcelain=v1 --untracked-files=all )
}

function Get-WorkingTreeSignature {
    $entries = @(
        Get-ChildItem -LiteralPath $repoPath -File -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\\.git(\\|$)' } |
            ForEach-Object {
                $relativePath = $_.FullName.Substring($repoPath.Length).ToLowerInvariant()
                "$relativePath|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
            } |
            Sort-Object
    )
    [string]::Join("`n", $entries)
}

$pendingSync = $true
$lastChange = Get-Date
$previousSignature = $null

while ($true) {
    Start-Sleep -Seconds $pollSeconds

    $currentSignature = Get-WorkingTreeSignature
    if ($null -eq $previousSignature) {
        $previousSignature = $currentSignature
    }
    elseif ($currentSignature -ne $previousSignature) {
        $previousSignature = $currentSignature
        $pendingSync = $true
        $lastChange = Get-Date
    }

    if ($pendingSync -and ((Get-Date) - $lastChange).TotalSeconds -ge $debounceSeconds) {
        $pendingSync = $false
        $status = Get-GitStatus

        if ($status.Count -gt 0) {
            try {
                Invoke-Git @('-c', 'core.autocrlf=false', 'add', '-f', '--all', '.')
                & git -C $repoPath diff --cached --quiet
                if ($LASTEXITCODE -ne 0) {
                    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                    Invoke-Git @('commit', '-m', "Auto-sync: $timestamp")
                    Invoke-Git @('push', 'origin', 'main')
                }
            }
            catch {
                $pendingSync = $true
                $lastChange = (Get-Date).AddSeconds(-$debounceSeconds + $retrySeconds)
            }
        }
    }
}
