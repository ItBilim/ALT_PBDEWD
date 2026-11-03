param([ValidateSet('Verify','Prepare','All','Aggregate','Tests','Seal')][string]$Mode='Verify')
$ErrorActionPreference='Stop'
$pythonPath=$env:PBDEWD_PYTHON
if(-not $pythonPath){
 $bundled=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
 if(Test-Path -LiteralPath $bundled){$pythonPath=$bundled}else{$pythonPath=(Get-Command python -ErrorAction Stop).Source}
}
Push-Location $PSScriptRoot
try{
 switch($Mode){
  'Tests' { & $pythonPath -B -X utf8 -m experiment.tests_run }
  'All' {
   & $pythonPath -B -X utf8 -m experiment.run --prepare
   if($LASTEXITCODE -ne 0){throw 'Feature rebuild failed'}
   & $pythonPath -B -X utf8 -m experiment.run --all
  }
  default { & $pythonPath -B -X utf8 -m experiment.run ('--'+$Mode.ToLowerInvariant()) }
 }
 if($LASTEXITCODE -ne 0){throw ('Experiment command failed: '+$Mode)}
}finally{Pop-Location}
