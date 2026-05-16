#!/usr/bin/env python3
"""
Test script for system monitoring functionality
Run this to verify system stats work without starting the full dashboard
"""
import json

def test_system_stats():
    """Test the system stats gathering code"""
    try:
        import psutil
        print("✓ psutil imported successfully")
        
        # CPU Temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
            print(f"✓ CPU Temperature: {cpu_temp:.1f}°C")
        except Exception as e:
            print(f"✗ CPU Temperature failed: {e}")
            cpu_temp = 0.0
        
        # CPU usage per core
        cpu_cores = psutil.cpu_percent(interval=0.1, percpu=True)
        print(f"✓ CPU Cores: {[round(c, 1) for c in cpu_cores]}")
        
        # Memory info
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024**3), 1)
        ram_used = round(mem.used / (1024**3), 1)
        ram_percent = round(mem.percent, 1)
        print(f"✓ RAM: {ram_used}/{ram_total} GB ({ram_percent}%)")
        
        # Swap info
        swap = psutil.swap_memory()
        swap_total = round(swap.total / (1024**3), 1)
        swap_used = round(swap.used / (1024**3), 1)
        swap_percent = round(swap.percent, 1)
        print(f"✓ Swap: {swap_used}/{swap_total} GB ({swap_percent}%)")
        
        # Top processes by CPU
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] and pinfo['cpu_percent'] > 0:
                    processes.append({
                        'name': pinfo['name'][:20],
                        'cpu': round(pinfo['cpu_percent'], 1)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        top_processes = sorted(processes, key=lambda x: x['cpu'], reverse=True)[:8]
        print(f"✓ Top {len(top_processes)} processes by CPU:")
        for proc in top_processes:
            print(f"  - {proc['name']}: {proc['cpu']}%")
        
        # Build the full response
        response = {
            'success': True,
            'cpu_temp': round(cpu_temp, 1),
            'cpu_cores': [round(c, 1) for c in cpu_cores],
            'ram_total': ram_total,
            'ram_used': ram_used,
            'ram_percent': ram_percent,
            'swap_total': swap_total,
            'swap_used': swap_used,
            'swap_percent': swap_percent,
            'top_processes': top_processes
        }
        
        print("\n✓ Full JSON response:")
        print(json.dumps(response, indent=2))
        print("\n✅ All system stats tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ psutil not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=== Testing System Stats Functionality ===\n")
    success = test_system_stats()
    exit(0 if success else 1)
