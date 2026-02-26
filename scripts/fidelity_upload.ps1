# Fidelity CSV Upload - PowerShell Script
#
# Uploads Fidelity position and activity CSVs to the CANSLIM Analyzer API.
#
# Usage:
#   .\fidelity_upload.ps1                          # Upload from Downloads folder
#   .\fidelity_upload.ps1 -Folder "C:\path\to\csvs" # Upload from specific folder
#   .\fidelity_upload.ps1 -Watch                    # Watch Downloads for new files
#   .\fidelity_upload.ps1 -Sync                     # Upload and sync to portfolio

param(
    [string]$Folder = "$env:USERPROFILE\Downloads",
    [string]$ApiUrl = "http://100.104.189.36:8001",
    [switch]$Watch,
    [switch]$Sync,
    [int]$Interval = 30
)

function Upload-FidelityFile {
    param(
        [string]$FilePath,
        [string]$Endpoint
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "  ERROR: File not found: $FilePath" -ForegroundColor Red
        return $null
    }

    $fileName = Split-Path $FilePath -Leaf
    Write-Host "  Uploading: $fileName" -ForegroundColor Cyan

    try {
        $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)
        $boundary = [System.Guid]::NewGuid().ToString()

        $bodyLines = @(
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
            "Content-Type: text/csv",
            "",
            [System.Text.Encoding]::UTF8.GetString($fileBytes),
            "--$boundary--"
        )
        $body = $bodyLines -join "`r`n"

        $response = Invoke-RestMethod -Uri "$ApiUrl$Endpoint" `
            -Method Post `
            -ContentType "multipart/form-data; boundary=$boundary" `
            -Body $body `
            -TimeoutSec 30

        return $response
    }
    catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

function Upload-Positions {
    param([string]$FilePath)

    $result = Upload-FidelityFile -FilePath $FilePath -Endpoint "/api/fidelity/upload-positions"
    if ($result) {
        Write-Host "  OK: $($result.positions_count) positions, `$$([math]::Round($result.total_value, 2)) total" -ForegroundColor Green
    }
    return $result
}

function Upload-Activity {
    param([string]$FilePath)

    $result = Upload-FidelityFile -FilePath $FilePath -Endpoint "/api/fidelity/upload-activity"
    if ($result) {
        Write-Host "  OK: $($result.new_trades) new trades, $($result.skipped_duplicates) duplicates skipped" -ForegroundColor Green
    }
    return $result
}

function Sync-ToPortfolio {
    Write-Host "  Syncing to portfolio..." -ForegroundColor Cyan
    try {
        $result = Invoke-RestMethod -Uri "$ApiUrl/api/fidelity/sync-to-portfolio" -Method Post -TimeoutSec 30
        Write-Host "  OK: $($result.added) added, $($result.updated) updated, $($result.removed) removed" -ForegroundColor Green
        return $result
    }
    catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Main
Write-Host ""
Write-Host "Fidelity CSV Upload" -ForegroundColor White
Write-Host "Folder: $Folder"
Write-Host "API: $ApiUrl"
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ("-" * 50)

if ($Watch) {
    Write-Host "Watching for new files (Ctrl+C to stop)..." -ForegroundColor Yellow
    $seen = @{}

    # Mark existing files as seen
    Get-ChildItem "$Folder\Portfolio_Positions_*.csv" -ErrorAction SilentlyContinue | ForEach-Object { $seen[$_.FullName] = $true }
    Get-ChildItem "$Folder\Accounts_History*.csv" -ErrorAction SilentlyContinue | ForEach-Object { $seen[$_.FullName] = $true }

    Write-Host "Tracking $($seen.Count) existing files.`n"

    while ($true) {
        Start-Sleep -Seconds $Interval

        Get-ChildItem "$Folder\Portfolio_Positions_*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
            if (-not $seen.ContainsKey($_.FullName)) {
                Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] New positions file!" -ForegroundColor Yellow
                $result = Upload-Positions -FilePath $_.FullName
                $seen[$_.FullName] = $true
                if ($Sync -and $result) { Sync-ToPortfolio }
            }
        }

        Get-ChildItem "$Folder\Accounts_History*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
            if (-not $seen.ContainsKey($_.FullName)) {
                Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] New activity file!" -ForegroundColor Yellow
                Upload-Activity -FilePath $_.FullName
                $seen[$_.FullName] = $true
            }
        }
    }
}
else {
    # One-shot: find and upload latest files
    $posFile = Get-ChildItem "$Folder\Portfolio_Positions_*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $actFile = Get-ChildItem "$Folder\Accounts_History*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($posFile) {
        $result = Upload-Positions -FilePath $posFile.FullName
    } else {
        Write-Host "  No positions CSV found (Portfolio_Positions_*.csv)" -ForegroundColor Yellow
    }

    if ($actFile) {
        Upload-Activity -FilePath $actFile.FullName
    } else {
        Write-Host "  No activity CSV found (Accounts_History*.csv)" -ForegroundColor Yellow
    }

    if ($Sync -and $posFile -and $result) {
        Sync-ToPortfolio
    }

    Write-Host ("-" * 50)
    Write-Host "Done.`n"
}
