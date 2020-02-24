doc_run_nogpu() {
    docker run \
        -it \
        -v ${PWD}:/repo \
        --workdir /repo \
        rubiks-img \
        /bin/bash
}

doc_run_nogpu
