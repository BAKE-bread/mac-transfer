
DEVICE_PATH="2-7"

echo "Attempting to reset USB device at $DEVICE_PATH"
echo "De-authorizing..."
echo '0' | sudo tee /sys/bus/usb/devices/$DEVICE_PATH/authorized

sleep 2 # 等待2秒确保设备已禁用

echo "Re-authorizing..."
echo '1' | sudo tee /sys/bus/usb/devices/$DEVICE_PATH/authorized
echo "Reset command sent."