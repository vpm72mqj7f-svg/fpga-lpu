#!/usr/bin/env python3
"""SCP the bringup project to ARM server (172.16.95.198)"""
import paramiko, os, sys, time
from scp import SCPClient

HOST = "172.16.95.198"
USER = "liyan"
PASS = "liyan"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_DIR = "/home/liyan/bringup"

EXCLUDE_DIRS = {".git", ".claude", "logs", "__pycache__"}
SKIP_EXT = {".exe", ".dll", ".jar", ".zip", ".pdf", ".ttf", ".gif", ".so", ".brd", ".DSN", ".docx"}

print(f"Connecting to {HOST}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15)
print("Connected.")

# Clean and pre-create all directories
print("Preparing remote directories...")
ssh.exec_command(f"rm -rf {REMOTE_DIR}")
time.sleep(1)

dirs_to_create = set()
for root, dirs, files in os.walk(LOCAL_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    rel = os.path.relpath(root, LOCAL_DIR).replace("\\", "/")
    if rel == ".":
        continue
    path_parts = rel.split("/")
    if "jre" in path_parts:
        continue
    dirs_to_create.add(os.path.join(REMOTE_DIR, rel).replace("\\", "/"))

for d in sorted(dirs_to_create):
    ssh.exec_command(f"mkdir -p {d}")

time.sleep(1)
print(f"  Created {len(dirs_to_create)} directories.")

# Collect files to transfer
files_to_send = []
skip_count = 0
for root, dirs, files in os.walk(LOCAL_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    rel_root = os.path.relpath(root, LOCAL_DIR).replace("\\", "/")
    path_parts = rel_root.split("/")

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in SKIP_EXT:
            skip_count += 1
            continue
        if "jre" in path_parts:
            continue
        local_path = os.path.join(root, f)
        remote_path = os.path.join(REMOTE_DIR, rel_root, f).replace("\\", "/")
        files_to_send.append((local_path, remote_path))

print(f"Transferring {len(files_to_send)} files ({skip_count} skipped)...")
with SCPClient(ssh.get_transport(), socket_timeout=30) as scp:
    for i, (local, remote) in enumerate(files_to_send):
        try:
            scp.put(local, remote)
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(files_to_send)}...")
        except Exception as e:
            print(f"  FAIL {local} -> {remote}: {e}")

# Verify
print("\nVerifying...")
stdin, stdout, stderr = ssh.exec_command("find ~/bringup -type f | head -30 && echo '...' && find ~/bringup -type f | wc -l")
ver = stdout.read().decode().strip()
print(ver)

ssh.close()
print("\nDone.")
