# Warning, don't run this script as root!

doc_run() {
    # This does many things.
    # First, it runs docker as a non-privelaged user,
    # with an interactive TTY session, and with all GPUs accessible.
    # It does so within the tensorflow 2.1.0 GPU image with python3,
    # and runs the python training script, with all the provided arguments.
    # It also links the directories for this repository,
    # and the NVIDIA Compute Cache (~/.nv).
    # NOTE: The first run many be very slow.
    #       This is due to CUDA filling up the Compute Cache.
    #       This file should persist under the '~/.nv' directory on the
    #       host machine, so later runs are faster.
    sudo docker run \
        -u $(id -u):$(id -g) \
        -d \
        --gpus all \
        -v /home/qt/.nv:/.nv \
        -v /home/qt/git/rubiks:/repo \
        --workdir /repo \
        --env CUDA_CACHE_PATH=/.nv \
        tensorflow/tensorflow:2.1.0-gpu-py3 \
        python rubiks.py $@
}

doc_run \
    --model-name res2-200-0001-50 \
    --epochs 150 \
    --learning-rate 0.0001 \
    --dropout-rate 0.50 \
    --batch-size 200

doc_run \
    --model-name res2-200-0001-10 \
    --epochs 150 \
    --learning-rate 0.0001 \
    --dropout-rate 0.10 \
    --batch-size 200

doc_run \
    --model-name res2-100-0005-50 \
    --epochs 150 \
    --learning-rate 0.0005 \
    --dropout-rate 0.50 \
    --batch-size 200

doc_run \
    --model-name res2-200-0005-10 \
    --epochs 150 \
    --learning-rate 0.0005 \
    --dropout-rate 0.10 \
    --batch-size 200
