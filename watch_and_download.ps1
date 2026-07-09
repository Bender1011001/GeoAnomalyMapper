<#
Watch a remote GeoAnomalyMapper scan and download a filtered results archive.

Configuration is supplied with parameters or environment variables; this script
does not embed SSH keys, hosts, ports, users, or remote paths.

Example with placeholder infrastructure:
  .\watch_and_download.ps1 `
    -SshKey "$env:USERPROFILE\.ssh\id_ed25519" `
    -RemoteUser "root" `
    -RemoteHost "gpu.example.com" `
    -Port 22 `
    -RemoteDir "/workspace/geo/data/biondi_exploration" `
    -RemoteArchive "/workspace/geo/results_archive.zip"

Equivalent environment variables:
  GEOANOMALY_SSH_KEY, GEOANOMALY_REMOTE_USER, GEOANOMALY_REMOTE_HOST,
  GEOANOMALY_REMOTE_PORT, GEOANOMALY_REMOTE_RESULTS_DIR,
  GEOANOMALY_REMOTE_ARCHIVE, GEOANOMALY_LOCAL_ARCHIVE,
  GEOANOMALY_POLL_SECONDS
#>

param(
    [string]$SshKey = $env:GEOANOMALY_SSH_KEY,
    [string]$RemoteUser = $env:GEOANOMALY_REMOTE_USER,
    [string]$RemoteHost = $env:GEOANOMALY_REMOTE_HOST,
    [int]$Port = 0,
    [string]$RemoteDir = $env:GEOANOMALY_REMOTE_RESULTS_DIR,
    [string]$RemoteArchive = $env:GEOANOMALY_REMOTE_ARCHIVE,
    [string]$LocalDest = $env:GEOANOMALY_LOCAL_ARCHIVE,
    [int]$PollSeconds = 0
)

$ErrorActionPreference = "Continue"

function Get-EnvInt {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Default
    )

    $raw = [Environment]::GetEnvironmentVariable($Name)
    $parsed = 0
    if (-not [string]::IsNullOrWhiteSpace($raw) -and [int]::TryParse($raw, [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Require-Value {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][AllowNull()][string]$Value,
        [Parameter(Mandatory = $true)][string]$EnvName
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Missing required setting '$Name'. Pass -$Name or set $EnvName."
    }
    return $Value
}

function ConvertTo-RemoteShellLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

if ($Port -le 0) {
    $Port = Get-EnvInt -Name "GEOANOMALY_REMOTE_PORT" -Default 22
}
if ($PollSeconds -le 0) {
    $PollSeconds = Get-EnvInt -Name "GEOANOMALY_POLL_SECONDS" -Default 60
}
if ([string]::IsNullOrWhiteSpace($LocalDest)) {
    $LocalDest = Join-Path $PSScriptRoot "results_archive.zip"
}

$SshKey = Require-Value -Name "SshKey" -Value $SshKey -EnvName "GEOANOMALY_SSH_KEY"
$RemoteUser = Require-Value -Name "RemoteUser" -Value $RemoteUser -EnvName "GEOANOMALY_REMOTE_USER"
$RemoteHost = Require-Value -Name "RemoteHost" -Value $RemoteHost -EnvName "GEOANOMALY_REMOTE_HOST"
$RemoteDir = Require-Value -Name "RemoteDir" -Value $RemoteDir -EnvName "GEOANOMALY_REMOTE_RESULTS_DIR"
$RemoteArchive = Require-Value -Name "RemoteArchive" -Value $RemoteArchive -EnvName "GEOANOMALY_REMOTE_ARCHIVE"

$RemoteEndpoint = "$RemoteUser@$RemoteHost"
$sshOptions = @("-p", "$Port", "-i", $SshKey, "-o", "StrictHostKeyChecking=no")
$scpOptions = @("-P", "$Port", "-i", $SshKey, "-o", "StrictHostKeyChecking=no")

Write-Host "Started watching remote scan process on $RemoteEndpoint..."

while ($true) {
    # Check if process is running.
    $processCheck = & ssh @sshOptions $RemoteEndpoint "ps aux | grep run_full_scan.py | grep -v grep" 2>$null
    
    if (-not $processCheck) {
        if ($LASTEXITCODE -eq 1) {
            Write-Host "Process finished! Zipping results..."
        } else {
            Write-Host "SSH error or hiccup (Exit Code $LASTEXITCODE). Retrying next poll..."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        # Zip the interesting outputs (JSONs, CSVs, reports, and small 3D models/images).
        # Exclude massive .npy arrays to save download time.
        $remoteDirLiteral = ConvertTo-RemoteShellLiteral -Value $RemoteDir
        $remoteArchiveLiteral = ConvertTo-RemoteShellLiteral -Value $RemoteArchive
        $zipCommand = "cd $remoteDirLiteral && zip -r $remoteArchiveLiteral . -i '*.json' '*.csv' '*.txt' '*.png' '*.stl' '*.vtk'"
        & ssh @sshOptions $RemoteEndpoint $zipCommand 2>$null

        Write-Host "Downloading archive to $LocalDest..."
        & scp @scpOptions "$($RemoteEndpoint):$RemoteArchive" $LocalDest

        Write-Host "Download complete!"

        try {
            Add-Type -AssemblyName System.Windows.Forms
            $global:balmsg = New-Object System.Windows.Forms.NotifyIcon
            $path = (Get-Process -id $pid).Path
            $balmsg.Icon = [System.Drawing.Icon]::ExtractAssociatedIcon($path)
            $balmsg.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
            $balmsg.BalloonTipText = "The GeoAnomalyMapper scan has finished and results have been downloaded to $LocalDest"
            $balmsg.BalloonTipTitle = "Scan Complete"
            $balmsg.Visible = $true
            $balmsg.ShowBalloonTip(10000)
        }
        catch {
            Write-Host "Windows notification could not be displayed: $($_.Exception.Message)"
        }

        break
    }

    Write-Host "Still running... Checking again in $PollSeconds seconds."
    Start-Sleep -Seconds $PollSeconds
}
