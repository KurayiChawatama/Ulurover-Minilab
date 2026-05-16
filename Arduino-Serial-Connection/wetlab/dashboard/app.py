 #!/usr/bin/env python3
"""
Minilab Dashboard - Web interface for the wetlab gas sensors, weather station
and Camera NoIR V2.
"""
import os
import subprocess
import glob
import csv
import serial
import time
import signal
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_SCRIPT = os.path.join(BASE_DIR, 'read-wetlab-csv.py')
DATA_DIR = BASE_DIR
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
PHOTOS_DIR = os.path.join(MEDIA_DIR, 'photos')
VIDEOS_DIR = os.path.join(MEDIA_DIR, 'videos')

# Weather station configuration
WEATHER_STATION_DIR = os.path.join(os.path.dirname(BASE_DIR), 'weather-station')
WEATHER_DATA_DIR = WEATHER_STATION_DIR

# Create media directories if they don't exist
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Wetlab live monitoring state
live_monitor_active = False
live_serial_connection = None

# Weather station monitoring state
weather_live_active = False
weather_serial_connection = None

# Camera streaming state
camera_stream_process = None
camera_recording_process = None


def get_color_camera_args():
    """Common camera args tuned for visible-light color rendering on NoIR modules."""
    return [
        '--awb', 'auto',
        '--metering', 'average',
        '--saturation', '1.1',
        '--contrast', '1.05'
    ]

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status for the wetlab gas sensors"""
    # Check for Arduino ports
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    
    # Get list of recent CSV files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'mq135_log_*.csv')), 
                      key=os.path.getmtime, reverse=True)[:5]
    csv_list = [os.path.basename(f) for f in csv_files]
    
    # Check if MQ135 is connected
    wetlab_connected = live_monitor_active and live_serial_connection is not None
    wetlab_port = live_serial_connection.port if wetlab_connected else None
    
    return jsonify({
        'success': True,
        'connected': wetlab_connected,
        'current_port': wetlab_port,
        'available_ports': ports,
        'recent_files': csv_list,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/run', methods=['POST'])
def run_collection():
    """Run data collection with specified parameters"""
    try:
        data = request.get_json()
        duration = int(data.get('duration', 10))
        
        # Auto-detect port for the wetlab Nano
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if not ports:
            return jsonify({'success': False, 'error': 'No Arduino port found'}), 400
        
        # Use first available Arduino port
        port = ports[0]
        
        # Generate output filename
        output_file = os.path.join(DATA_DIR, f"mq135_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Run data collection
        print(f"Starting data collection for {duration} seconds...")
        cmd = [
            'python', PYTHON_SCRIPT,
            '--port', port,
            '--seconds', str(duration),
            '--output', output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
        
        if result.returncode != 0:
            return jsonify({
                'success': False, 
                'error': 'Data collection failed',
                'details': result.stderr
            }), 500
        
        return jsonify({
            'success': True,
            'output_file': os.path.basename(output_file),
            'message': f'Data collected for {duration}s',
            'log': result.stdout
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Data collection timeout'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/<filename>', methods=['GET'])
def get_data(filename):
    """Get CSV data for plotting"""
    try:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        data = {
            'timestamps': [],
            'sensors': {}
        }
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Initialize sensor data arrays
            for header in headers:
                if header != 'Seconds':
                    data['sensors'][header] = []
            
            # Read all rows
            for row in reader:
                data['timestamps'].append(float(row['Seconds']))
                for header in headers:
                    if header != 'Seconds':
                        try:
                            data['sensors'][header].append(float(row[header]))
                        except (ValueError, KeyError):
                            data['sensors'][header].append(None)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'data': data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/start', methods=['POST'])
def start_live_monitor():
    """Start live monitoring"""
    global live_monitor_active, live_serial_connection
    
    try:
        if live_monitor_active:
            return jsonify({'success': False, 'error': 'Live monitor already running'}), 400
        
        # Find Arduino port - single Arduino setup
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if not ports:
            return jsonify({'success': False, 'error': 'No Arduino port found'}), 400
        
        # Use first available Arduino port
        port = ports[0]
        
        # Open serial connection (Arduino should already be programmed)
        live_serial_connection = serial.Serial(port, 9600, timeout=2)
        live_monitor_active = True
        
        # Flush any initial garbage data
        time.sleep(0.5)
        live_serial_connection.reset_input_buffer()
        
        return jsonify({'success': True, 'message': 'Live monitoring started', 'port': port})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/stop', methods=['POST'])
def stop_live_monitor():
    """Stop live monitoring"""
    global live_monitor_active, live_serial_connection
    
    try:
        if live_serial_connection:
            live_serial_connection.close()
            live_serial_connection = None
        
        live_monitor_active = False
        
        return jsonify({'success': True, 'message': 'Live monitoring stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/data', methods=['GET'])
def get_live_data():
    """Get current live sensor readings"""
    global live_monitor_active, live_serial_connection
    
    if not live_monitor_active or not live_serial_connection:
        return jsonify({'success': False, 'error': 'Live monitor not running'}), 400
    
    try:
        # Read a line from serial
        if live_serial_connection.in_waiting > 0:
            line = live_serial_connection.readline().decode('utf-8', errors='ignore').strip()
            
            if line and ',' in line and not line.lower().startswith('seconds'):
                parts = line.split(',')
                
                if len(parts) >= 2:
                    return jsonify({
                        'success': True,
                        'timestamp': float(parts[0]),
                        'sensors': {
                            f'CO2_PPM_{i+1}': float(parts[i+1]) 
                            for i in range(len(parts) - 1)
                        }
                    })
        
        return jsonify({'success': True, 'data': None})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== SERVO CONTROL ENDPOINTS =====

# Servo state tracking
servo_running = False
servo_serial_connection = None

@app.route('/api/servo/start', methods=['POST'])
def servo_start():
    """Start rotating the servo drum"""
    global servo_running, servo_serial_connection
    
    try:
        # If not connected, establish connection
        if servo_serial_connection is None:
            ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            if not ports:
                return jsonify({'success': False, 'error': 'No Arduino port found'}), 400
            
            port = ports[0]
            servo_serial_connection = serial.Serial(port, 9600, timeout=1)
            time.sleep(0.5)
            servo_serial_connection.reset_input_buffer()
        
        # Send start command to Arduino
        servo_serial_connection.write(b'SERVO_START\n')
        servo_running = True
        
        # Read confirmation
        time.sleep(0.2)
        if servo_serial_connection.in_waiting > 0:
            response = servo_serial_connection.readline().decode('utf-8', errors='ignore').strip()
            return jsonify({
                'success': True,
                'message': 'Servo rotation started',
                'response': response
            })
        
        return jsonify({'success': True, 'message': 'Servo rotation started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/servo/stop', methods=['POST'])
def servo_stop():
    """Stop rotating the servo drum"""
    global servo_running, servo_serial_connection
    
    try:
        if servo_serial_connection is None:
            return jsonify({'success': False, 'error': 'Servo not connected'}), 400
        
        # Send stop command to Arduino
        servo_serial_connection.write(b'SERVO_STOP\n')
        servo_running = False
        
        # Read confirmation
        time.sleep(0.2)
        if servo_serial_connection.in_waiting > 0:
            response = servo_serial_connection.readline().decode('utf-8', errors='ignore').strip()
            return jsonify({
                'success': True,
                'message': 'Servo rotation stopped',
                'response': response
            })
        
        return jsonify({'success': True, 'message': 'Servo rotation stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/servo/status', methods=['GET'])
def servo_status():
    """Get servo status"""
    return jsonify({
        'success': True,
        'running': servo_running,
        'connected': servo_serial_connection is not None
    })


@app.route('/api/servo/angle', methods=['POST'])
def servo_set_angle():
    """Set servo angle (kept for compatibility). For continuous servos this maps to speed/direction."""
    global servo_serial_connection

    try:
        data = request.get_json() or {}
        angle = int(data.get('angle', -1))

        if angle < 0 or angle > 180:
            return jsonify({'success': False, 'error': 'Angle must be 0-180'}), 400

        # Ensure serial connection
        if servo_serial_connection is None:
            ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            if not ports:
                return jsonify({'success': False, 'error': 'No Arduino port found'}), 400

            port = ports[0]
            servo_serial_connection = serial.Serial(port, 9600, timeout=1)
            time.sleep(0.5)
            servo_serial_connection.reset_input_buffer()

        # Send angle command (Arduino maps this to pulse-width / speed)
        cmd = f"SERVO_ANGLE:{angle}\n".encode('utf-8')
        servo_serial_connection.write(cmd)

        # Read confirmation if available
        time.sleep(0.1)
        response = None
        if servo_serial_connection.in_waiting > 0:
            response = servo_serial_connection.readline().decode('utf-8', errors='ignore').strip()

        return jsonify({'success': True, 'angle': angle, 'response': response})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== CAMERA ENDPOINTS =====

@app.route('/api/camera/photo', methods=['POST'])
def capture_photo():
    """Capture a still photo using rpicam-still"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'photo_{timestamp}.jpg'
        filepath = os.path.join(PHOTOS_DIR, filename)
        
        # Use rpicam-still for photo capture (newer command)
        # -n or --nopreview: no preview window
        # -o: output file
        # -t: allow brief settle time so AWB/exposure can stabilize
        cmd = [
            'rpicam-still',
            '-n',  # nopreview
            '-o', filepath,
            '-t', '800',
            '--width', '1920',
            '--height', '1080'
        ]
        cmd.extend(get_color_camera_args())
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': 'Photo capture failed',
                'details': result.stderr
            }), 500
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'Photo file was not created'
            }), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/api/camera/media/photos/{filename}',
            'message': 'Photo captured successfully'
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Camera timeout'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/video/start', methods=['POST'])
def start_video_recording():
    """Start video recording using rpicam-vid"""
    global camera_recording_process
    
    try:
        if camera_recording_process and camera_recording_process.poll() is None:
            return jsonify({'success': False, 'error': 'Recording already in progress'}), 400
        
        data = request.get_json()
        duration = int(data.get('duration', 10))  # Duration in seconds
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'video_{timestamp}.h264'
        filepath = os.path.join(VIDEOS_DIR, filename)
        
        # Use rpicam-vid for video recording
        # -n: nopreview
        # -t: duration in milliseconds
        # -o: output file
        cmd = [
            'rpicam-vid',
            '-n',  # nopreview
            '-o', filepath,
            '-t', str(duration * 1000),  # Convert to milliseconds
            '--width', '1920',
            '--height', '1080',
            '--framerate', '30'
        ]
        cmd.extend(get_color_camera_args())
        
        camera_recording_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'duration': duration,
            'message': f'Video recording started for {duration} seconds'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/video/status', methods=['GET'])
def video_recording_status():
    """Check if video recording is in progress"""
    global camera_recording_process
    
    if camera_recording_process and camera_recording_process.poll() is None:
        return jsonify({'recording': True})
    else:
        return jsonify({'recording': False})

@app.route('/api/camera/stream')
def camera_stream():
    """Stream camera feed as MJPEG using continuous still captures"""
    def generate_frames():
        """Generate MJPEG frames by capturing stills continuously"""
        import tempfile
        
        while True:
            try:
                # Create a temporary file for each frame
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Capture a single frame quickly
                cmd = [
                    'rpicam-still',
                    '-n',  # No preview
                    '-o', tmp_path,
                    '-t', '220',  # Give AWB/exposure time to settle
                    '--width', '640',
                    '--height', '480',
                    '--quality', '80'  # Lower quality for faster streaming
                ]
                cmd.extend(get_color_camera_args())
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=2
                )
                
                if result.returncode == 0 and os.path.exists(tmp_path):
                    # Read the image file
                    with open(tmp_path, 'rb') as f:
                        frame = f.read()
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Yield frame in MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    # Clean up temp file on error
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Stream error: {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                time.sleep(0.1)
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/media/<media_type>/<filename>')
def serve_media(media_type, filename):
    """Serve captured photos or videos"""
    try:
        if media_type == 'photos':
            filepath = os.path.join(PHOTOS_DIR, filename)
        elif media_type == 'videos':
            filepath = os.path.join(VIDEOS_DIR, filename)
        else:
            return jsonify({'error': 'Invalid media type'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/media/list', methods=['GET'])
def list_media():
    """List all captured photos and videos"""
    try:
        photos = sorted(
            glob.glob(os.path.join(PHOTOS_DIR, '*.jpg')),
            key=os.path.getmtime,
            reverse=True
        )
        videos = sorted(
            glob.glob(os.path.join(VIDEOS_DIR, '*.h264')),
            key=os.path.getmtime,
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'photos': [
                {
                    'filename': os.path.basename(p),
                    'url': f'/api/camera/media/photos/{os.path.basename(p)}',
                    'timestamp': os.path.getmtime(p),
                    'size': os.path.getsize(p)
                }
                for p in photos
            ],
            'videos': [
                {
                    'filename': os.path.basename(v),
                    'url': f'/api/camera/media/videos/{os.path.basename(v)}',
                    'timestamp': os.path.getmtime(v),
                    'size': os.path.getsize(v)
                }
                for v in videos
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/media/delete', methods=['POST'])
def delete_media():
    """Delete a photo or video"""
    try:
        data = request.get_json()
        media_type = data.get('type')
        filename = data.get('filename')
        
        if media_type == 'photo':
            filepath = os.path.join(PHOTOS_DIR, filename)
        elif media_type == 'video':
            filepath = os.path.join(VIDEOS_DIR, filename)
        else:
            return jsonify({'success': False, 'error': 'Invalid media type'}), 400
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': 'File deleted'})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== SYSTEM MONITORING ENDPOINT =====

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """Get Raspberry Pi system statistics"""
    try:
        import psutil
        
        # CPU Temperature (Raspberry Pi specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0
        except Exception as e:
            print(f"Warning: Could not read CPU temperature: {e}")
            cpu_temp = 0.0
        
        # CPU usage per core
        cpu_cores = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory info
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024**3), 1)
        ram_used = round(mem.used / (1024**3), 1)
        ram_percent = round(mem.percent, 1)
        
        # Swap info
        swap = psutil.swap_memory()
        swap_total = round(swap.total / (1024**3), 1)
        swap_used = round(swap.used / (1024**3), 1)
        swap_percent = round(swap.percent, 1)
        
        # Top processes by CPU
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] and pinfo['cpu_percent'] > 0:
                    processes.append({
                        'name': pinfo['name'][:20],  # Truncate long names
                        'cpu': round(pinfo['cpu_percent'], 1)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage and get top 8
        top_processes = sorted(processes, key=lambda x: x['cpu'], reverse=True)[:8]
        
        print(f"System stats: temp={cpu_temp}°C, RAM={ram_used}/{ram_total}GB, processes={len(top_processes)}")
        
        return jsonify({
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
        })
        
    except ImportError as e:
        print(f"ERROR: psutil not available: {e}")
        return jsonify({'success': False, 'error': f'psutil not installed: {str(e)}'}), 500
    except Exception as e:
        print(f"ERROR in get_system_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== WEATHER STATION ENDPOINTS =====
# Note: All CO2 values are stored and returned as raw sensor readings.
# A +400 ppm offset is applied in the frontend to account for atmospheric baseline.

@app.route('/api/weather/status', methods=['GET'])
def get_weather_status():
    """Get weather station system status"""
    # Check for available ports
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    
    # Get list of recent weather CSV files
    csv_files = sorted(glob.glob(os.path.join(WEATHER_DATA_DIR, 'weather_log_*.csv')), 
                      key=os.path.getmtime, reverse=True)[:5]
    csv_list = [os.path.basename(f) for f in csv_files]
    
    return jsonify({
        'success': True,
        'available_ports': ports,
        'recent_files': csv_list,
        'connected': weather_live_active,
        'current_port': weather_serial_connection.port if weather_serial_connection else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/weather/live/start', methods=['POST'])
def start_weather_live_monitor():
    """Start live weather monitoring on ttyACM0"""
    global weather_live_active, weather_serial_connection
    
    try:
        if weather_live_active:
            return jsonify({'success': False, 'error': 'Weather monitor already running'}), 400
        
        # Weather station is on ttyACM0
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if not ports:
            return jsonify({'success': False, 'error': 'No Arduino port found'}), 400
        
        # Use ttyACM0 for weather station
        port = '/dev/ttyACM0' if '/dev/ttyACM0' in ports else ports[0]
        
        # Open serial connection (9600 baud for weather station)
        weather_serial_connection = serial.Serial(port, 9600, timeout=2)
        weather_live_active = True
        
        # Flush initial data
        time.sleep(0.5)
        weather_serial_connection.reset_input_buffer()
        
        return jsonify({'success': True, 'message': 'Weather monitoring started', 'port': port})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/weather/live/stop', methods=['POST'])
def stop_weather_live_monitor():
    """Stop live weather monitoring"""
    global weather_live_active, weather_serial_connection
    
    try:
        if weather_serial_connection:
            weather_serial_connection.close()
            weather_serial_connection = None
        
        weather_live_active = False
        
        return jsonify({'success': True, 'message': 'Weather monitoring stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/weather/live/data', methods=['GET'])
def get_weather_live_data():
    """Get current live weather readings
    
    Note: CO2 values returned are raw sensor readings.
    The frontend applies a +400 ppm offset to account for 
    atmospheric baseline CO2 concentration.
    """
    global weather_live_active, weather_serial_connection
    
    if not weather_live_active or not weather_serial_connection:
        return jsonify({'success': False, 'error': 'Weather monitor not running'}), 400
    
    try:
        if weather_serial_connection.in_waiting > 0:
            line = weather_serial_connection.readline().decode('utf-8', errors='ignore').strip()
            
            # Skip system messages and header line
            if line.startswith('SYSTEM') or line.startswith('RTC:') or \
               line.startswith('BME280:') or line.startswith('VEML6070:') or \
               line.startswith('SETUP') or 'Date' in line or line.startswith('SD:'):
                return jsonify({'success': True, 'data': None})
            
            # Parse data: Date,Time,CO2,CH4,H2,Temp,Pressure,Humidity,UV
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 9:
                    try:
                        return jsonify({
                            'success': True,
                            'date': parts[0],
                            'time': parts[1],
                            'co2': float(parts[2]),  # Raw sensor value
                            'ch4': float(parts[3]),
                            'h2': float(parts[4]),
                            'temperature': float(parts[5]),
                            'pressure': float(parts[6]),
                            'humidity': float(parts[7]),
                            'uv': int(parts[8]),
                            'timestamp': datetime.now().isoformat()
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing weather data: {e}, line: {line}")
                        return jsonify({'success': True, 'data': None})
        
        return jsonify({'success': True, 'data': None})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/weather/record/start', methods=['POST'])
def start_weather_recording():
    """Start recording weather data to CSV file"""
    try:
        data = request.get_json()
        duration = int(data.get('duration', 60))
        
        # Generate output filename
        output_file = os.path.join(WEATHER_DATA_DIR, 
                                   f"weather_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Weather station is on ttyACM0
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if not ports:
            return jsonify({'success': False, 'error': 'No Arduino port found'}), 400
        
        port = '/dev/ttyACM0' if '/dev/ttyACM0' in ports else ports[0]
        
        # Open serial and start recording
        with open(output_file, 'w', newline='') as csvfile:
            csvfile.write('Date,Time,CO2,CH4,H2,Temp,Pressure,Humidity,UV\n')
            
            with serial.Serial(port, 9600, timeout=2) as ser:
                time.sleep(1)
                ser.reset_input_buffer()
                
                start_time = time.time()
                while (time.time() - start_time) < duration:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        # Skip system messages
                        if line and not any(x in line for x in ['SYSTEM', 'RTC:', 'BME280:', 'VEML6070:', 'SETUP', 'Date']):
                            if ',' in line:
                                csvfile.write(line + '\n')
                                csvfile.flush()
                    
                    time.sleep(0.1)
        
        return jsonify({
            'success': True,
            'output_file': os.path.basename(output_file),
            'message': f'Weather data recorded for {duration}s',
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/weather/data/<filename>', methods=['GET'])
def get_weather_data(filename):
    """Get weather CSV data for plotting
    
    Note: CO2 values returned are raw sensor readings.
    Apply +400 ppm offset in the frontend for atmospheric baseline.
    """
    try:
        filepath = os.path.join(WEATHER_DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        data = {
            'date': [],
            'time': [],
            'co2': [],
            'ch4': [],
            'h2': [],
            'temperature': [],
            'pressure': [],
            'humidity': [],
            'uv': []
        }
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    data['date'].append(row['Date'])
                    data['time'].append(row['Time'])
                    data['co2'].append(float(row['CO2']))
                    data['ch4'].append(float(row['CH4']))
                    data['h2'].append(float(row['H2']))
                    data['temperature'].append(float(row['Temp']))
                    data['pressure'].append(float(row['Pressure']))
                    data['humidity'].append(float(row['Humidity']))
                    data['uv'].append(int(row['UV']))
                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to error: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'filename': filename,
            'data': data,
            'record_count': len(data['date'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== PORT MANAGEMENT =====

import threading
port_lock = threading.Lock()

@app.route('/api/ports/list', methods=['GET'])
def list_all_ports():
    """List all available ports and their status"""
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    
    port_info = []
    for port in ports:
        info = {
            'port': port,
            'mq135_active': live_monitor_active and live_serial_connection and live_serial_connection.port == port,
            'weather_active': weather_live_active and weather_serial_connection and weather_serial_connection.port == port
        }
        port_info.append(info)
    
    return jsonify({
        'success': True,
        'ports': port_info,
        'total_ports': len(ports)
    })

@app.route('/api/ports/restart', methods=['POST'])
def restart_port():
    """Restart a specific serial port or all ports"""
    global live_serial_connection, live_monitor_active
    global weather_serial_connection, weather_live_active
    
    try:
        data = request.get_json()
        port = data.get('port', 'all')
        
        with port_lock:
            if port == 'all':
                # Restart all connections
                if live_serial_connection:
                    live_serial_connection.close()
                    live_serial_connection = None
                    live_monitor_active = False
                
                if weather_serial_connection:
                    weather_serial_connection.close()
                    weather_serial_connection = None
                    weather_live_active = False
                
                time.sleep(1)
                
                return jsonify({
                    'success': True,
                    'message': 'All ports restarted successfully'
                })
            else:
                # Restart specific port
                if live_serial_connection and live_serial_connection.port == port:
                    live_serial_connection.close()
                    time.sleep(0.5)
                    live_serial_connection = serial.Serial(port, 9600, timeout=2)
                    time.sleep(0.5)
                    live_serial_connection.reset_input_buffer()
                    
                if weather_serial_connection and weather_serial_connection.port == port:
                    weather_serial_connection.close()
                    time.sleep(0.5)
                    weather_serial_connection = serial.Serial(port, 9600, timeout=2)
                    time.sleep(0.5)
                    weather_serial_connection.reset_input_buffer()
                
                return jsonify({
                    'success': True,
                    'message': f'Port {port} restarted successfully',
                    'port': port
                })
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== SERVER CONTROL =====

@app.route('/api/server/shutdown', methods=['POST'])
def shutdown_server():
    """Gracefully shutdown the Flask server"""
    try:
        # Close all serial connections
        global live_serial_connection, weather_serial_connection
        
        if live_serial_connection:
            live_serial_connection.close()
        
        if weather_serial_connection:
            weather_serial_connection.close()
        
        # Stop camera processes if running
        global camera_stream_process, camera_recording_process
        
        if camera_stream_process:
            camera_stream_process.terminate()
            camera_stream_process.wait()
        
        if camera_recording_process:
            camera_recording_process.terminate()
            camera_recording_process.wait()
        
        print("Dashboard server shutting down...")
        
        # Use os._exit to forcefully exit (works in debug mode)
        def shutdown():
            os._exit(0)
        
        # Schedule shutdown after response is sent
        import threading
        threading.Timer(0.5, shutdown).start()
        
        return jsonify({
            'success': True,
            'message': 'Server shutting down...'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Run on all interfaces so it can be accessed from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
