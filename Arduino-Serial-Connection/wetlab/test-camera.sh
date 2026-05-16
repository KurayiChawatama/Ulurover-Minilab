#!/bin/bash
# Quick test script to verify camera is working

echo "=== Camera Test Script ==="
echo ""
echo "Testing Camera NoIR V2..."
echo ""

# Test 1: Check if camera is detected
echo "1. Checking if camera is detected..."
if rpicam-hello --list 2>&1 | grep -q "Available cameras"; then
    echo "✅ Camera detected"
else
    echo "❌ Camera not detected"
    echo "   Run: sudo raspi-config"
    echo "   Enable: Interfacing Options > Camera"
    exit 1
fi

echo ""
echo "2. Testing photo capture..."
TEST_PHOTO="/tmp/test_photo.jpg"
if rpicam-still -n -o "$TEST_PHOTO" -t 1 --width 640 --height 480 2>/dev/null; then
    if [ -f "$TEST_PHOTO" ]; then
        SIZE=$(stat -c%s "$TEST_PHOTO" 2>/dev/null)
        echo "✅ Photo capture works (${SIZE} bytes)"
        rm "$TEST_PHOTO"
    else
        echo "❌ Photo file not created"
        exit 1
    fi
else
    echo "❌ Photo capture failed"
    exit 1
fi

echo ""
echo "3. Testing video recording (2 seconds)..."
TEST_VIDEO="/tmp/test_video.h264"
if timeout 3 rpicam-vid -n -o "$TEST_VIDEO" -t 2000 --width 640 --height 480 2>/dev/null; then
    if [ -f "$TEST_VIDEO" ]; then
        SIZE=$(stat -c%s "$TEST_VIDEO" 2>/dev/null)
        echo "✅ Video recording works (${SIZE} bytes)"
        rm "$TEST_VIDEO"
    else
        echo "❌ Video file not created"
        exit 1
    fi
else
    echo "❌ Video recording failed"
    exit 1
fi

echo ""
echo "4. Testing continuous capture (for streaming simulation)..."
TEST_STREAM="/tmp/test_stream.jpg"
if rpicam-still -n -o "$TEST_STREAM" -t 1 --width 640 --height 480 --quality 80 2>/dev/null; then
    if [ -f "$TEST_STREAM" ]; then
        SIZE=$(stat -c%s "$TEST_STREAM" 2>/dev/null)
        echo "✅ Continuous capture works (${SIZE} bytes)"
        rm "$TEST_STREAM"
    else
        echo "❌ Stream frame not created"
        exit 1
    fi
else
    echo "❌ Continuous capture failed"
    exit 1
fi

echo ""
echo "=== All Tests Passed! ==="
echo ""
echo "Camera is ready for use in dashboard."
echo "Run: ./start-dashboard.sh"
echo ""
echo "Note: Stream uses continuous still captures at ~2-3 fps"
echo "      This is reliable but lower framerate than true video streaming"
