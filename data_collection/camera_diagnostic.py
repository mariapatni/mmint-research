import pyrealsense2 as rs
import numpy as np

def print_device_info(device):
    """Print comprehensive device information"""
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    
    info_types = [
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.firmware_version,
        rs.camera_info.product_line,
        rs.camera_info.usb_type_descriptor,
        rs.camera_info.product_id,
        rs.camera_info.physical_port,
        rs.camera_info.debug_op_code,
        rs.camera_info.advanced_mode,
        rs.camera_info.asic_serial_number,
        rs.camera_info.firmware_update_id,
    ]
    
    for info_type in info_types:
        try:
            value = device.get_info(info_type)
            print(f"{info_type}: {value}")
        except Exception as e:
            print(f"{info_type}: Error - {e}")

def print_sensor_info(device):
    """Print information about all sensors"""
    print("\n" + "=" * 50)
    print("SENSOR INFORMATION")
    print("=" * 50)
    
    sensors = device.query_sensors()
    print(f"Number of sensors found: {len(sensors)}")
    
    for i, sensor in enumerate(sensors):
        print(f"\n--- Sensor {i+1} ---")
        
        # Basic sensor info
        try:
            print(f"Name: {sensor.get_info(rs.camera_info.name)}")
        except:
            print("Name: Not available")
            
        try:
            print(f"Serial Number: {sensor.get_info(rs.camera_info.serial_number)}")
        except:
            print("Serial Number: Not available")
            
        try:
            print(f"Firmware Version: {sensor.get_info(rs.camera_info.firmware_version)}")
        except:
            print("Firmware Version: Not available")
        
        # Check what streams this sensor supports
        print("Supported streams:")
        for stream_type in [rs.stream.depth, rs.stream.color, rs.stream.infrared, rs.stream.gyro, rs.stream.accel]:
            try:
                profiles = sensor.get_stream_profiles()
                for profile in profiles:
                    if profile.stream_type() == stream_type:
                        video_profile = rs.video_stream_profile(profile)
                        print(f"  - {stream_type}: {video_profile.width()}x{video_profile.height()} @ {profile.fps()}fps")
            except:
                pass

def check_stream_capabilities():
    """Check what streams can be enabled"""
    print("\n" + "=" * 50)
    print("STREAM CAPABILITY TEST")
    print("=" * 50)
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    for device in devices:
        print(f"\nTesting device: {device.get_info(rs.camera_info.name)}")
        
        # Test different stream configurations
        configs_to_test = [
            (rs.stream.depth, 640, 480, rs.format.z16, 30),
            (rs.stream.color, 640, 480, rs.format.bgr8, 30),
            (rs.stream.infrared, 640, 480, rs.format.y8, 30),
        ]
        
        for stream_type, width, height, format_type, fps in configs_to_test:
            try:
                config = rs.config()
                config.enable_stream(stream_type, width, height, format_type, fps)
                
                pipeline = rs.pipeline()
                profile = pipeline.start(config)
                
                print(f"  ✓ {stream_type}: {width}x{height} @ {fps}fps - SUCCESS")
                
                pipeline.stop()
                
            except Exception as e:
                print(f"  ✗ {stream_type}: {width}x{height} @ {fps}fps - FAILED: {e}")

if __name__ == "__main__":
    try:
        # Create a context
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found!")
            exit(1)
        
        print(f"Found {len(devices)} RealSense device(s)")
        
        for i, device in enumerate(devices):
            print(f"\n{'='*60}")
            print(f"DEVICE {i+1}")
            print(f"{'='*60}")
            print_device_info(device)
            print_sensor_info(device)
        
        check_stream_capabilities()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 