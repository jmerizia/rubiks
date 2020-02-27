doc_run_gpu() {
    docker run \
        -it \
        --gpus all \
        -v ${PWD}:/repo \
        --workdir /repo \
        rubiks-img \
        /bin/bash
}

doc_run_nogpu() {
    docker run \
        -it \
        -v ${PWD}:/repo \
        --workdir /repo \
        rubiks-img \
        /bin/bash
}

if
    [[ "$1" == "nogpu" ]]
    then doc_run_nogpu
    else doc_run_gpu
fi
