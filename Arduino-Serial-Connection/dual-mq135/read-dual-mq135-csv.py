import serial
import csv
import time
import argparse
import glob
from datetime import datetime

# Find all Arduino serial ports (ttyUSB* or ttyACM*)
def find_arduino_port():
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if not ports:
        print("No Arduino serial ports found.")
        exit(1)
    return ports[0]

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Read CO2 PPM from the wetlab Nano with multiple MQ-135 sensors and log to CSV.")
    parser.add_argument('--port', type=str, default=None, help='Serial port (default: auto-detect)')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--seconds', type=int, default=0, help='Number of seconds to run (0 = run indefinitely)')
    parser.add_argument('--output', type=str, default=None, help='CSV output file (default: mq135_log_<timestamp>.csv)')
    return parser.parse_args()

def main():
    args = parse_args()
    port = args.port if args.port else find_arduino_port()
    baud = args.baud
    seconds = args.seconds
    if args.output:
        csv_file = args.output
    else:
        csv_file = f"mq135_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        ser = serial.Serial(port, baud, timeout=2)
        print(f"Connected to {port} at {baud} baud.")
        print("Reading from wetlab Nano with multiple MQ-135 sensors...")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # First line will be the header from Arduino, we'll capture it
        header_captured = False
        start_time = time.time()
        elapsed = 0
        try:
            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue
                    
                if not header_captured:
                    # Capture the header line from Arduino
                    if line.lower().startswith('seconds'):
                        writer.writerow(line.split(','))
                        header_captured = True
                        continue
                        
                if line and ',' in line:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2:  # At least Seconds and first sensor reading
                            writer.writerow(parts)
                            print(line)
                    except Exception:
                        pass
                        
                elapsed = time.time() - start_time
                if seconds > 0 and elapsed >= seconds:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
    print(f"CSV saved to {csv_file}")

if __name__ == "__main__":
    main()
