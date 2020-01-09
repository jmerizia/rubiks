# naming convention:
# {model type}-{learning rate}-{batch size}-{dropout rate}

python rubiks.py                  \
    --model-name res2-200-0001-50 \
    --epochs 120                  \
    --learning-rate 0.0001        \
    --dropout-rate 0.50           \
    --batch-size 200 &
P1=$!

python rubiks.py                  \
    --model-name res2-200-0001-10 \
    --epochs 120                  \
    --learning-rate 0.0001        \
    --dropout-rate 0.10           \
    --batch-size 200 &
P2=$!

python rubiks.py                  \
    --model-name res2-100-0005-50 \
    --epochs 120                  \
    --learning-rate 0.0005        \
    --dropout-rate 0.50           \
    --batch-size 200 &
P3=$!

python rubiks.py                  \
    --model-name res2-200-0005-10 \
    --epochs 120                  \
    --learning-rate 0.0005        \
    --dropout-rate 0.10           \
    --batch-size 200 &
P4=$!

time wait $P1 $P2 $P3 $P4
