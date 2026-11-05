import os
import sys
import subprocess
import time
import json
from pathlib import Path

# 1. Environment & Path Validation Setup
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

def log(msg, status="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{status}] {msg}")

def verify_environment():
    if not VENV_PYTHON.exists():
        log(f"Virtual environment python not found at: {VENV_PYTHON}", "ERROR")
        log("Please ensure a '.venv' folder exists in the execution directory.", "ERROR")
        sys.exit(1)
    log(f"Found virtual environment python: {VENV_PYTHON}")

def check_windows_os_gpu_errors():
    """Queries Windows System Event Logs for actual GPU/Display/NVIDIA driver errors and parses detailed fault parameters."""
    log("Checking Windows OS Event Logs for real GPU/Driver errors and possible driver error details...")
    ps_script = """
    try {
        # Query nvlddmkm and Display errors from System log
        $events = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName=@('nvlddmkm', 'Display')} -MaxEvents 15 -ErrorAction SilentlyContinue
        if ($events) {
            $results = foreach ($e in $events) {
                if ($e.Level -le 2) {
                    # Extract XML properties or string details to get deep driver error context
                    $xml = [xml]$e.ToXml()
                    $data = $xml.Event.EventData.Data
                    
                    [PSCustomObject]@{
                        Time = $e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss.fff')
                        Id = $e.Id
                        Level = $e.LevelDisplayName
                        Source = $e.ProviderName
                        Message = ($e.Message -replace "`r`n"," ").Trim()
                        RawData = $data -join ' | '
                    }
                }
            }
            if ($results) {
                if ($results -is [array]) {
                    $results | ConvertTo-Json -Depth 3 -Compress
                } else {
                    @($results) | ConvertTo-Json -Depth 3 -Compress
                }
            }
        }
    } catch {
        # Catch and ignore if logs aren't accessible
    }
    """
    try:
        result = subprocess.run(["powershell", "-Command", ps_script], capture_output=True, text=True, timeout=8)
        if result.returncode == 0 and result.stdout.strip():
            raw_out = result.stdout.strip()
            parsed = json.loads(raw_out)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
    except Exception as e:
        log(f"Could not query OS event logs: {e}", "WARN")
    return []

# 2. Worker Payload Script (Only real workloads, no simulated errors)
WORKER_CODE = """
import torch
import sys
import time
import json
import traceback

def report_gpus():
    try:
        if not torch.cuda.is_available():
            print(json.dumps({"error": "No CUDA devices available."}))
            sys.exit(10)
        
        gpus = []
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            })
        print(json.dumps({"gpus": gpus}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({
            "error": "OS/Driver Error during GPU discovery",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(11)

def run_load_test(gpu_id):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Initializing heavy workload on GPU {gpu_id}...", flush=True)
        
        n = 16000
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        c = torch.randn(n, n, device=device)
        
        start_time = time.time()
        iterations = 0
        target_duration = 45 # 45 seconds heavy load
        
        while time.time() - start_time < target_duration:
            res = torch.matmul(a, b)
            res = torch.addmm(res, b, c)
            a = torch.sin(res)
            torch.cuda.synchronize(device)
            iterations += 1
            
        print(json.dumps({
            "status": "SUCCESS",
            "iterations": iterations,
            "duration_sec": round(time.time() - start_time, 2),
            "matrix_dim": f"{n}x{n}"
        }))
    except Exception as e:
        print(json.dumps({
            "status": "ERROR",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(20)

if __name__ == "__main__":
    mode = sys.argv[1]
    g_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    if mode == "list":
        report_gpus()
    elif mode == "load":
        run_load_test(g_id)
"""

def run_worker(*args, timeout=None):
    cmd = [str(VENV_PYTHON), "-c", WORKER_CODE] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -999, e.stdout or "", "Timeout Expired"

def select_gpu(gpus):
    if len(gpus) == 1:
        log(f"Only 1 GPU detected. Automatically targeting GPU 0: {gpus[0]['name']}")
        return gpus[0]
    
    print("\n--- Available GPUs ---")
    for gpu in gpus:
        print(f"  [{gpu['id']}] {gpu['name']} ({gpu['memory_total_gb']} GB VRAM)")
    print("----------------------\n")
    
    while True:
        try:
            user_input = input(f"Select GPU ID to test [0-{len(gpus)-1}] (default 0): ").strip()
            choice = 0 if not user_input else int(user_input)
            
            selected = next((g for g in gpus if g["id"] == choice), None)
            if selected:
                log(f"Selected GPU [{selected['id']}]: {selected['name']}")
                return selected
            print("Invalid GPU ID selected. Please try again.")
        except ValueError:
            print("Please enter a valid integer ID.")

def generate_reports(report_data):
    json_path = PROJECT_ROOT / "gpu_test_report.json"
    md_path = PROJECT_ROOT / "gpu_test_report.md"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
        
    target_gpu = report_data["target_gpu"]
    os_errors = report_data["os_level_driver_errors"]
    
    md_content = [
        "# GPU PyTorch & OS-Level Driver Diagnostics Report",
        f"**Timestamp:** {report_data['timestamp']}",
        f"**Target GPU:** ID {target_gpu['id']} - {target_gpu['name']} ({target_gpu['memory_total_gb']} GB VRAM)",
        f"**OS-Level Driver Status:** {'⚠️ REAL DRIVER ERRORS FOUND IN WINDOWS EVENT LOG' if os_errors else '✅ NO OS-LEVEL DRIVER ERRORS DETECTED'}\n",
        "## Test Execution Summary",
        "| Test Phase | Status | Exit Code | Notes |",
        "|---|---|---|---|"
    ]
    
    for test in report_data["tests"]:
        notes = test.get("details", {}).get("message") or test.get("details", {}).get("iterations", "N/A")
        if isinstance(notes, int):
            notes = f"{notes} heavy matrix iterations"
        md_content.append(f"| {test['name']} | {test['status']} | {test['exit_code']} | {notes} |")
        
    md_content.append("\n## Windows OS-Level Driver Error Details (Event Logs)")
    if os_errors:
        for err in os_errors:
            md_content.append(f"### ❌ [{err.get('Source')} - Event ID {err.get('Id')}] at `{err.get('Time')}`")
            md_content.append(f"- **Message:** {err.get('Message')}")
            if err.get('RawData'):
                md_content.append(f"- **Driver Error Parameters / Context:** `{err.get('RawData')}`")
    else:
        md_content.append("- Clean: No display/driver errors recorded in Windows Event Viewer.")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    log(f"JSON report saved to: {json_path}")
    log(f"Markdown report saved to: {md_path}")

def main():
    verify_environment()
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_gpu": None,
        "tests": [],
        "os_level_driver_errors": []
    }
    
    log("=== Step 1: Querying Available GPUs ===")
    code, out, err = run_worker("list")
    if code != 0:
        log(f"GPU Discovery failed: {err or out}", "ERROR")
        sys.exit(1)
        
    try:
        data = json.loads(out.strip().splitlines()[-1])
        gpus = data.get("gpus", [])
    except Exception as e:
        log(f"Parsing output failed: {out}", "ERROR")
        sys.exit(1)
        
    selected_gpu = select_gpu(gpus)
    report_data["target_gpu"] = selected_gpu
    gpu_id = selected_gpu["id"]

    # TEST 1: Heavy Normal Load Test
    log("=== Step 2: Running Heavy Matrix Performance Load Test (45s) ===")
    start_t = time.time()
    code, out, err = run_worker("load", str(gpu_id))
    duration = time.time() - start_t
    
    test1_record = {"name": "Heavy Matrix Load & Normal Exit", "exit_code": code, "duration_sec": round(duration, 2)}
    try:
        res_json = json.loads(out.strip().splitlines()[-1])
        test1_record["status"] = "PASSED" if code == 0 and res_json.get("status") == "SUCCESS" else "FAILED"
        test1_record["details"] = res_json
    except Exception:
        test1_record["status"] = "FAILED"
        test1_record["details"] = {"raw_output": out}
    report_data["tests"].append(test1_record)
    log(f"Normal Test Finished with status [{test1_record['status']}]")

    # TEST 2: Kill Load Test (Abrupt Termination Test)
    log("=== Step 3: Running Abrupt Process Termination (Kill) Test ===")
    process = subprocess.Popen([str(VENV_PYTHON), "-c", WORKER_CODE, "load", str(gpu_id)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    log("Letting workload run for 10 seconds before terminating process abruptly...")
    time.sleep(10)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        
    test3_record = {
        "name": "Process Kill Simulation",
        "exit_code": process.returncode,
        "status": "PASSED" if process.returncode != 0 else "WARNING",
        "details": {"message": f"Process terminated successfully with exit code {process.returncode}"}
    }
    report_data["tests"].append(test3_record)
    log(f"Kill Test Executed with Exit Code: {process.returncode}", "SUCCESS")

    # Post-test OS check for real hardware/driver errors
    log("=== Step 4: Scanning Windows Event Logs for Real Driver Errors & Parameter Context ===")
    report_data["os_level_driver_errors"] = check_windows_os_gpu_errors()

    # Step 5: Report Generation
    log("=== Step 5: Generating Final Diagnostics Reports ===")
    generate_reports(report_data)
    log("=== All verification routines completed successfully! ===")

if __name__ == "__main__":
    main()