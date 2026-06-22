[CmdletBinding()]
param(
    [string]$CommitMessage = "Backstop commit at session end",
    [string]$RepoRoot
)

# Auto-push safety net for THIS project (Codex Stop hook). Mirrors the Claude
# Vault's .codex/hooks/session-end-git-backstop.ps1: atomic Win32 CreateDirectory
# lock (interlocks with the bash auto-push on the same .vault.lock path), commit
# leftover changes EXCLUDING .claude (Claude owns it), then push. Never throws
# into the session.

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDirectory "..\..")).ProviderPath
}
$SafeDirectory = $null

function Write-HookWarning {
    param([string]$Message)

    [Console]::Error.WriteLine("[session-end-git-backstop] $Message")
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [switch]$AllowFailure
    )

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $gitArguments = @("-c", "safe.directory=$SafeDirectory") + $Arguments
        $output = & git @gitArguments 2>&1 |
            ForEach-Object { $_.ToString() } |
            Where-Object { $_ -ne "System.Management.Automation.RemoteException" }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }

    if ($exitCode -ne 0 -and -not $AllowFailure) {
        $joinedOutput = ($output -join [Environment]::NewLine).Trim()
        if ($joinedOutput.Length -gt 0) {
            throw "git $($Arguments -join ' ') failed with exit code ${exitCode}: $joinedOutput"
        }

        throw "git $($Arguments -join ' ') failed with exit code ${exitCode}."
    }

    [PSCustomObject]@{
        ExitCode = $exitCode
        Output = @($output)
    }
}

function Acquire-VaultLock {
    # Atomic mutex shared with the Claude bash hook (which uses `mkdir`). NOTE:
    # PowerShell's `New-Item -ItemType Directory` is NOT atomic across processes -
    # it pre-checks existence then calls the idempotent .NET Directory.CreateDirectory,
    # so two processes can both "win". Use the native Win32 CreateDirectory, which
    # fails (ERROR_ALREADY_EXISTS) if the directory exists: exactly one caller wins,
    # and it interlocks with bash `mkdir` on the same .vault.lock path.
    param(
        [Parameter(Mandatory = $true)][string]$LockPath,
        [int]$MaxTries = 40,
        [int]$StaleSeconds = 120
    )
    if (-not ([System.Management.Automation.PSTypeName]'Win32.VaultLock').Type) {
        Add-Type -Namespace Win32 -Name VaultLock -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError=true, CharSet=System.Runtime.InteropServices.CharSet.Unicode)]
public static extern bool CreateDirectory(string lpPathName, System.IntPtr lpSecurityAttributes);
'@
    }
    for ($i = 0; $i -lt $MaxTries; $i++) {
        if ([Win32.VaultLock]::CreateDirectory($LockPath, [System.IntPtr]::Zero)) {
            "codex-backstop pid=$PID $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
                Out-File -FilePath (Join-Path $LockPath 'owner') -Encoding ascii -ErrorAction SilentlyContinue
            return $true
        }
        # Reclaim a stale lock left behind by a crashed run.
        if (Test-Path -LiteralPath $LockPath) {
            $age = (Get-Date) - (Get-Item -LiteralPath $LockPath -Force).LastWriteTime
            if ($age.TotalSeconds -ge $StaleSeconds) {
                Remove-Item -LiteralPath $LockPath -Recurse -Force -ErrorAction SilentlyContinue
                continue
            }
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Release-VaultLock {
    param([string]$LockPath)
    if ($LockPath -and (Test-Path -LiteralPath $LockPath)) {
        Remove-Item -LiteralPath $LockPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$VaultLockPath = $null
$VaultLockHeld = $false

try {
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath
    $SafeDirectory = $RepoRoot -replace "\\", "/"
    Set-Location -LiteralPath $RepoRoot

    $repoResult = Invoke-Git -Arguments @("rev-parse", "--show-toplevel") -AllowFailure
    if ($repoResult.ExitCode -ne 0 -or $repoResult.Output.Count -eq 0) {
        $details = (($repoResult.Output | Where-Object { $_ }) -join [Environment]::NewLine).Trim()
        Write-HookWarning "Could not identify a Git repository at '$RepoRoot'. $details"
        exit 1
    }

    $repoRoot = $repoResult.Output[-1].Trim()
    $SafeDirectory = $repoRoot -replace "\\", "/"
    Set-Location -LiteralPath $repoRoot

    # Acquire the shared commit lock before touching git. If another instance
    # (Claude or Codex) holds it, it is committing/pushing already - back off.
    $VaultLockPath = Join-Path $repoRoot ".vault.lock"
    if (-not (Acquire-VaultLock -LockPath $VaultLockPath)) {
        exit 0
    }
    $VaultLockHeld = $true

    $upstreamResult = Invoke-Git -Arguments @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") -AllowFailure
    if ($upstreamResult.ExitCode -ne 0 -or $upstreamResult.Output.Count -eq 0 -or [string]::IsNullOrWhiteSpace($upstreamResult.Output[-1])) {
        Write-HookWarning "No upstream is configured for the current branch in '$repoRoot'. Set one with: git push -u origin HEAD"
        exit 1
    }

    $upstream = $upstreamResult.Output[-1].Trim()

    $statusResult = Invoke-Git -Arguments @("status", "--porcelain", "--", ".", ":(exclude).claude")
    $dirtyLines = @($statusResult.Output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

    if ($dirtyLines.Count -gt 0) {
        Invoke-Git -Arguments @("add", "-A", "--", ".", ":(exclude).claude") | Out-Null

        $stagedResult = Invoke-Git -Arguments @("diff", "--cached", "--name-only", "--", ".", ":(exclude).claude")
        $stagedFiles = @($stagedResult.Output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

        if ($stagedFiles.Count -gt 0) {
            Invoke-Git -Arguments @("commit", "-m", $CommitMessage, "--", ".", ":(exclude).claude") | Out-Null
        }
    }

    $aheadResult = Invoke-Git -Arguments @("rev-list", "--count", "$upstream..HEAD") -AllowFailure
    if ($aheadResult.ExitCode -ne 0 -or $aheadResult.Output.Count -eq 0) {
        $details = (($aheadResult.Output | Where-Object { $_ }) -join [Environment]::NewLine).Trim()
        Write-HookWarning "Could not determine whether local commits need pushing to '$upstream'. $details"
        exit 1
    }

    $aheadText = $aheadResult.Output[-1].Trim()
    $aheadCount = 0
    if (-not [int]::TryParse($aheadText, [ref]$aheadCount)) {
        Write-HookWarning "Could not parse ahead count '$aheadText' for upstream '$upstream'."
        exit 1
    }

    if ($aheadCount -eq 0) {
        exit 0
    }

    $pushResult = Invoke-Git -Arguments @("push") -AllowFailure
    if ($pushResult.ExitCode -ne 0) {
        $details = (($pushResult.Output | Where-Object { $_ }) -join [Environment]::NewLine).Trim()
        Write-HookWarning "git push failed for '$repoRoot' to '$upstream'. $details"
        exit 1
    }

    exit 0
}
catch {
    Write-HookWarning $_.Exception.Message
    exit 1
}
finally {
    if ($VaultLockHeld) { Release-VaultLock -LockPath $VaultLockPath }
}
