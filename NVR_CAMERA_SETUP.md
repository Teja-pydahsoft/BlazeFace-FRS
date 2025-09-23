# NVR Camera Setup Guide for BlazeFace-FRS

## Overview
The BlazeFace-FRS system now supports multiple camera sources including NVR cameras, IP cameras, and video files.

## Supported Camera Types

### 1. Webcam (Default)
- **Source**: `"0"` (or camera index number)
- **Example**: `"0"`, `"1"`, `"2"`

### 2. NVR Camera (RTSP Stream)
- **Source**: `"rtsp://username:password@ip:port/path"`
- **Example**: `"rtsp://admin:password123@192.168.1.100:554/stream1"`

### 3. IP Camera (HTTP Stream)
- **Source**: `"http://ip:port/video"`
- **Example**: `"http://192.168.1.101:8080/video"`

### 4. Video File
- **Source**: `"/path/to/video.mp4"`
- **Example**: `"test_video.mp4"`

## Configuration

### Update `app/config.json`:

```json
{
    "camera_sources": {
        "webcam": "0",
        "nvr_camera": "rtsp://admin:password@192.168.1.100:554/stream1",
        "ip_camera": "http://192.168.1.101:8080/video",
        "video_file": "test_video.mp4"
    }
}
```

## NVR Camera Setup Steps

### 1. Find Your NVR Camera Details
- **IP Address**: Usually found in NVR settings (e.g., 192.168.1.100)
- **Port**: Default is usually 554 for RTSP
- **Username/Password**: Admin credentials for the camera
- **Stream Path**: Usually `/stream1` or `/ch1/stream1`

### 2. Common NVR Camera Formats
```
rtsp://admin:password@192.168.1.100:554/stream1
rtsp://admin:password@192.168.1.100:554/ch1/stream1
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### 3. Test Your RTSP Stream
You can test your RTSP stream using VLC or ffmpeg:
```bash
# Using VLC
vlc rtsp://admin:password@192.168.1.100:554/stream1

# Using ffmpeg
ffmpeg -i rtsp://admin:password@192.168.1.100:554/stream1 -t 10 test_output.mp4
```

## Using the Camera Selection

### In the Attendance Marking Interface:
1. Open **Attendance Marking** dialog
2. In the **Settings** section, find **Camera Source** dropdown
3. Select your desired camera source:
   - **webcam**: Use local webcam
   - **nvr_camera**: Use NVR camera (RTSP)
   - **ip_camera**: Use IP camera (HTTP)
   - **video_file**: Use video file
4. Click **Switch** button to change camera
5. Start attendance system

## Troubleshooting

### RTSP Connection Issues:
- Check IP address and port
- Verify username/password
- Ensure camera is accessible on network
- Check firewall settings

### Performance Issues:
- Reduce stream resolution in camera settings
- Use lower bitrate streams
- Check network bandwidth

### Common Error Messages:
- **"Failed to open camera source"**: Check connection details
- **"Camera not initialized"**: Verify camera is online and accessible
- **"Timeout error"**: Network connectivity issue

## Network Requirements

### For NVR/IP Cameras:
- **Bandwidth**: Minimum 2 Mbps per camera
- **Latency**: < 200ms for real-time processing
- **Network**: Stable connection recommended

### Port Requirements:
- **RTSP**: Port 554 (default)
- **HTTP**: Port 80, 8080, or custom port
- **HTTPS**: Port 443

## Security Considerations

### For Production Use:
- Change default passwords
- Use secure networks (VPN if remote)
- Enable camera authentication
- Regular firmware updates

### Example Secure Configuration:
```json
{
    "camera_sources": {
        "webcam": "0",
        "main_entrance": "rtsp://secure_user:strong_password@192.168.1.100:554/stream1",
        "backup_camera": "rtsp://secure_user:strong_password@192.168.1.101:554/stream1"
    }
}
```

## Advanced Configuration

### Multiple Camera Support:
You can configure multiple cameras and switch between them:

```json
{
    "camera_sources": {
        "webcam": "0",
        "entrance_camera": "rtsp://admin:pass@192.168.1.100:554/stream1",
        "lobby_camera": "rtsp://admin:pass@192.168.1.101:554/stream1",
        "classroom_camera": "rtsp://admin:pass@192.168.1.102:554/stream1"
    }
}
```

### Video File Testing:
Use video files to test the system without live cameras:
```json
{
    "camera_sources": {
        "test_video": "test_attendance.mp4",
        "sample_class": "classroom_recording.avi"
    }
}
```

## Performance Tips

1. **Use appropriate stream quality**: Balance between quality and performance
2. **Test network stability**: Ensure consistent connection
3. **Monitor system resources**: CPU and memory usage
4. **Use wired connections**: More stable than WiFi for critical applications

## Support

If you encounter issues:
1. Check camera accessibility from your computer
2. Verify network connectivity
3. Test with VLC or similar tools
4. Check system logs for detailed error messages
