export CUDA_VISIBLE_DEVICES=0
img_dir="./data/teleop_aloha"
python teleop_aloha.py \
    --output_dir $img_dir \
    --multi_asset

# export CUDA_VISIBLE_DEVICES=0
# img_dir="./data/viewer"
# python viewer.py \
#     --output_dir $img_dir \
#     --multi_asset \
#     -s /home/agilex/workspace/MIRROR/gaussian-splatting/data/vpx-room_v0 \
#     -m /home/agilex/workspace/MIRROR/gaussian-splatting/output/vpx-room_v0 \
#     --resolution 2

# export CUDA_VISIBLE_DEVICES=0
# img_dir="data/teleop_hand"
# python teleop_hand.py \
#     --output_dir $img_dir \
#     --multi_asset

# Teleop camera
# lsusb
# sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U241-if00-port0
# python teleop_real.py
