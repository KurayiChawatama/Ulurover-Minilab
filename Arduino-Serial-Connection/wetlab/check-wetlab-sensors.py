#!/usr/bin/env python3
"""
Diagnostic script to check raw analog values and voltages from MQ-135 sensors
This helps diagnose if sensors are getting different voltages
"""

import serial
import time
import glob

def find_arduino_port():
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if not ports:
        print("No Arduino serial ports found.")
        exit(1)
    return ports[0]

def main():
    port = find_arduino_port()
    
    try:
        ser = serial.Serial(port, 9600, timeout=2)
        print(f"Connected to {port}")
        print("Monitoring sensor voltages and raw values...")
        print("\nFormat: Seconds, CO2_PPM_B1, CO2_PPM_B2, CO2_PPM_B3")
        print("=" * 70)
        
        start_time = time.time()
        
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and ',' in line:
                    elapsed = int(time.time() - start_time)
                    print(f"{elapsed:3d}s: {line}")
                    
                    # Parse and analyze
                    if not line.lower().startswith('seconds'):
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                seconds = float(parts[0])
                                b1 = float(parts[1])
                                b2 = float(parts[2])
                                b3 = float(parts[3])
                                
                                # Check for significant differences
                                max_diff = max(abs(b1-b2), abs(b2-b3), abs(b1-b3))
                                if max_diff > 50:  # More than 50 PPM difference
                                    print(f"     ⚠️  Large variance detected: {max_diff:.1f} PPM difference")
                                    print(f"     Possible voltage issue - check power connections")
                            except ValueError:
                                pass
                                
            except KeyboardInterrupt:
                print("\n\nDiagnostic Summary:")
                print("If you see consistent differences between sensors (e.g., B1 always ~4 PPM higher than B3),")
                print("this suggests:")
                print("  1. Different sensor characteristics (normal - each sensor is unique)")
                print("  2. Voltage differences (check power distribution)")
                print("  3. Different environmental exposure (check sensor placement)")
                break
            except Exception:
                pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
