# Register Forecast ML scheduled tasks. Run as Administrator.
#
# This script registers tasks for the production weather forecasting
# project. Phase 2 only registers the hourly METAR logger; later phases
# will add the daily aggregator, retrain, forecast runner, and
# verification jobs.
#
# Re-run this script after pulling new code; existing tasks with the
# same name are overwritten.

#Requires -RunAsAdministrator

param(
    [string]$ProjectRoot = "D:\Projects\CO_Springs_Weather_ML",
    [string]$PythonExe = "D:\Python313\python.exe"
)

$ErrorActionPreference = "Stop"

function Register-MLTask {
    param(
        [string]$Name,
        [string]$Description,
        [string]$ScriptPath,
        $Trigger
    )

    Write-Host "Registering task: $Name"

    $action = New-ScheduledTaskAction `
        -Execute $PythonExe `
        -Argument "`"$ScriptPath`"" `
        -WorkingDirectory $ProjectRoot

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -DontStopIfGoingOnBatteries `
        -AllowStartIfOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
        -MultipleInstances IgnoreNew

    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel Limited

    $task = New-ScheduledTask `
        -Action $action `
        -Trigger $Trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description

    Register-ScheduledTask `
        -TaskName $Name `
        -InputObject $task `
        -Force | Out-Null
}

# --- Phase 2 task: hourly METAR logger ---

# Trigger at every hour, :05 past the hour. Repetition fires once an hour
# starting from the next :05 minute.
$loggerStart = (Get-Date).Date.AddHours((Get-Date).Hour + 1).AddMinutes(5)
# Note: Task Scheduler rejects [TimeSpan]::MaxValue. Use 10 years (3650 days)
# as the repetition duration -- effectively forever for our purposes.
$loggerTrigger = New-ScheduledTaskTrigger -Once -At $loggerStart `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

Register-MLTask `
    -Name "Forecast ML Hourly METAR" `
    -Description "Pull latest NWS observations for the 23-station forecasting network into forecast_ml.db" `
    -ScriptPath (Join-Path $ProjectRoot "scripts\metar_ml_logger.py") `
    -Trigger $loggerTrigger

# --- Phase 1 follow-up: daily aggregator at 00:30 MST ---

$aggregatorTrigger = New-ScheduledTaskTrigger -Daily -At "12:30AM"

Register-MLTask `
    -Name "Forecast ML Daily Aggregator" `
    -Description "Aggregate metar_hourly into daily_summary for the 23-station network" `
    -ScriptPath (Join-Path $ProjectRoot "scripts\daily_aggregator.py") `
    -Trigger $aggregatorTrigger

# --- Phase 5 task: nightly retrain at 02:00 MST ---

$retrainTrigger = New-ScheduledTaskTrigger -Daily -At "2:00AM"

Register-MLTask `
    -Name "Forecast ML Retrain" `
    -Description "Refit all 28 forecasting heads in production mode; updates is_active model_runs row" `
    -ScriptPath (Join-Path $ProjectRoot "scripts\retrain.py") `
    -Trigger $retrainTrigger

# --- Phase 4 task: daily forecast runner at 06:30 MST ---

$forecastTrigger = New-ScheduledTaskTrigger -Daily -At "6:30AM"

Register-MLTask `
    -Name "Forecast ML Daily Run" `
    -Description "Generate daily ML weather forecast for KCOS h=1..7 with NWS comparison; writes production_forecast + markdown report" `
    -ScriptPath (Join-Path $ProjectRoot "scripts\forecast_runner.py") `
    -Trigger $forecastTrigger

# --- Phase 5 task: verification at 23:55 MST ---

$verificationTrigger = New-ScheduledTaskTrigger -Daily -At "11:55PM"

Register-MLTask `
    -Name "Forecast ML Verification" `
    -Description "Score every issued forecast against the actual outcome once it lands" `
    -ScriptPath (Join-Path $ProjectRoot "scripts\verification_logger.py") `
    -Trigger $verificationTrigger

Write-Host ""
Write-Host "Done. All 5 production tasks registered:"
Write-Host "  - Forecast ML Hourly METAR        (every hour at :05)"
Write-Host "  - Forecast ML Daily Aggregator    (00:30 MST daily)"
Write-Host "  - Forecast ML Retrain             (02:00 MST daily)"
Write-Host "  - Forecast ML Daily Run           (06:30 MST daily)"
Write-Host "  - Forecast ML Verification        (23:55 MST daily)"
Write-Host ""
Write-Host "Verify all:     Get-ScheduledTask -TaskName 'Forecast ML*'"
Write-Host "Run on demand:  Start-ScheduledTask -TaskName 'Forecast ML Daily Run'"
Write-Host "Hourly logs:    $ProjectRoot\data\metar_ml_logger.log"
Write-Host "Daily reports:  $ProjectRoot\reports\YYYY-MM-DD.md"
Write-Host "Dashboard:      python $ProjectRoot\scripts\accuracy_dashboard.py"
