
#!/bin/bash

# export PATH=/mnt/bn/pico-mr-hl-taohu/anaconda3/bin:$PATH
# source activate conda
# source activate ex4d

# sudo apt-get install libgl1-mesa-glx -y

python infer.py --color_video examples/flower/render_180.mp4 \
                --mask_video examples/flower/mask_180.mp4 \
                --output_video outputs/flower.mp4
