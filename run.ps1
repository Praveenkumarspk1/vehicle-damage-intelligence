param(
    [ValidateSet("ui", "api", "test", "eval", "train")]
    [string]$Mode = "ui"
)

$ErrorActionPreference = "Stop"
python run.py --mode $Mode

