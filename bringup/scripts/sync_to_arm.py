#!/usr/bin/env python3
"""Sync bringup project to ARM server via SFTP"""
import paramiko, os, sys, stat as statmod

HOST = "172.16.95.198"
USER = "liyan"
PASS = "liyan"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_DIR = "/home/liyan/bringup"

SKIP = {".git", ".claude", "logs", "__pycache__", "jre"}
SKIP_EXT = {".exe", ".dll", ".jar", ".zip", ".pdf", ".ttf", ".gif", ".so", ".brd", ".DSN", ".docx", ".xls"}

def mkdirs(sftp, path):
    """Create remote directory recursively."""
    parts = path.replace("\\", "/").strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except (FileNotFoundError, IOError, OSError):
            try:
                sftp.mkdir(cur)
            except (IOError, OSError):
                pass

print(f"Connecting to {HOST}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15, banner_timeout=15)
print("Connected. Collecting files...")

# Collect all files first
jobs = []
for root, dirs, files in os.walk(LOCAL_DIR):
    dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
    rel = os.path.relpath(root, LOCAL_DIR).replace("\\", "/")
    parts = rel.split("/")
    if set(parts) & SKIP:
        continue
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in SKIP_EXT:
            continue
        local = os.path.join(root, f)
        remote = REMOTE_DIR + "/" + rel + "/" + f if rel != "." else REMOTE_DIR + "/" + f
        remote = remote.replace("\\", "/")
        jobs.append((local, remote))

print(f"Files to send: {len(jobs)}")

# Clean remote
print("Cleaning remote...")
stdin, stdout, stderr = ssh.exec_command(f"rm -rf {REMOTE_DIR}; mkdir -p {REMOTE_DIR}")
stdout.channel.recv_exit_status()

# Transfer via SFTP
sftp = ssh.open_sftp()
sent = 0
failed = 0
for local, remote in jobs:
    try:
        remote_dir = os.path.dirname(remote).replace("\\", "/")
        mkdirs(sftp, remote_dir)
        sftp.put(local, remote)
        sent += 1
        if sent % 50 == 0:
            print(f"  {sent}/{len(jobs)}...")
    except Exception as e:
        print(f"  FAIL {os.path.basename(local)}: {e}")
        failed += 1

sftp.close()

# Verify
print(f"\nDone: {sent} sent, {failed} failed")
stdin, stdout, stderr = ssh.exec_command(f"find {REMOTE_DIR} -type f | wc -l && ls {REMOTE_DIR}/rtl/ && ls {REMOTE_DIR}/scripts/")
print(stdout.read().decode())

ssh.close()
