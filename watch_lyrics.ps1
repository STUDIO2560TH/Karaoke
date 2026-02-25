
# Watch Lyrics Folder — Auto Commit & Push
# Run this script and it will watch the lyrics/ folder.
# When you add a .txt file, it auto commits and pushes to GitHub.

$repoPath = $PSScriptRoot
$lyricsPath = Join-Path $repoPath "lyrics"

# Create lyrics folder if it doesn't exist
if (-not (Test-Path $lyricsPath)) {
    New-Item -ItemType Directory -Path $lyricsPath | Out-Null
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Karaoke Lyrics Watcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Watching: $lyricsPath" -ForegroundColor Yellow
Write-Host "Drop a .txt file into the lyrics/ folder and it will auto push to GitHub!" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $lyricsPath
$watcher.Filter = "*.txt"
$watcher.EnableRaisingEvents = $true
$watcher.IncludeSubdirectories = $false

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $name = $Event.SourceEventArgs.Name

    # Small delay to let the file finish writing
    Start-Sleep -Seconds 2

    if (Test-Path $path) {
        Write-Host ""
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] New lyrics file detected: $name" -ForegroundColor Green

        Push-Location $using:repoPath
        try {
            git add "lyrics/$name"
            git commit -m "Add lyrics: $name"
            git push

            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pushed to GitHub! The sync action will start shortly." -ForegroundColor Cyan
        }
        catch {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ERROR: $($_.Exception.Message)" -ForegroundColor Red
        }
        Pop-Location
    }
}

Register-ObjectEvent $watcher "Created" -Action $action | Out-Null

# Keep script running
try {
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "`nWatcher stopped." -ForegroundColor Yellow
}
