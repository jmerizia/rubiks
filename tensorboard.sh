sudo docker run \
    -u $(id -u):$(id -g) \
    -it \
    --gpus all \
    -v /home/qt/.nv:/.nv \
    -v /home/qt/git/rubiks:/repo \
    --workdir /repo \
    -p 127.0.0.1:6006:6006/tcp \
    --env CUDA_CACHE_PATH=/.nv \
    tensorflow/tensorflow:2.1.0-gpu-py3 \
    tensorboard \
        --bind_all \
        --logdir test_logs
