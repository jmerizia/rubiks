# naming convention:
# {model type}-{learning rate}-{batch size}-{dropout rate}

python rubiks.py                 \
    --model-name res-100-0005-00 \
    --epochs 120                 \
    --learning-rate 0.0005       \
    --dropout-rate 0.00          \
    --batch-size 100 &

python rubiks.py                 \
    --model-name res-200-0005-00 \
    --epochs 120                 \
    --learning-rate 0.0005       \
    --dropout-rate 0.00          \
    --batch-size 200 &

python rubiks.py                 \
    --model-name res-100-0005-10 \
    --epochs 120                 \
    --learning-rate 0.0005       \
    --dropout-rate 0.10          \
    --batch-size 100 &

python rubiks.py                 \
    --model-name res-200-0005-10 \
    --epochs 120                 \
    --learning-rate 0.0005       \
    --dropout-rate 0.10          \
    --batch-size 200 &
