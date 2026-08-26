$ErrorActionPreference = 'Stop'

$projectPath = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$repoPath = $projectPath
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

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $repoPath
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [IO.NotifyFilters]::FileName -bor [IO.NotifyFilters]::DirectoryName -bor [IO.NotifyFilters]::LastWrite -bor [IO.NotifyFilters]::Size
$pendingSync = $true
$lastChange = Get-Date

$onChange = {
    param($sender, $eventArgs)
    $changedPath = $eventArgs.FullPath
    if ($changedPath -notmatch '\\.git(\\|$)') {
        $script:pendingSync = $true
        $script:lastChange = Get-Date
    }
}

$watcher.add_Changed($onChange)
$watcher.add_Created($onChange)
$watcher.add_Deleted($onChange)
$watcher.add_Renamed($onChange)
$watcher.EnableRaisingEvents = $true

try {
    while ($true) {
        Start-Sleep -Seconds 5

        if (-not $pendingSync) {
            continue
        }

        if (((Get-Date) - $lastChange).TotalSeconds -lt $debounceSeconds) {
            continue
        }

        $pendingSync = $false
        $status = Get-GitStatus
        if ($status.Count -eq 0) {
            continue
        }

        try {
            Invoke-Git @('-c', 'core.autocrlf=false', 'add', '-f', '--all', '.')
            & git -C $repoPath diff --cached --quiet
            if ($LASTEXITCODE -eq 0) {
                continue
            }

            $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            Invoke-Git @('commit', '-m', "Auto-sync: $timestamp")
            Invoke-Git @('push', 'origin', 'main')
        }
        catch {
            $pendingSync = $true
            $lastChange = (Get-Date).AddSeconds(-$debounceSeconds + $retrySeconds)
            Start-Sleep -Seconds $retrySeconds
        }
    }
}
finally {
    $watcher.remove_Changed($onChange)
    $watcher.remove_Created($onChange)
    $watcher.remove_Deleted($onChange)
    $watcher.remove_Renamed($onChange)
    $watcher.Dispose()
}
