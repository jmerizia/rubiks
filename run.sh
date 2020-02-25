doc_run_nogpu() {
    docker run \
        -it \
	--gpus all \
        -v ${PWD}:/repo \
        --workdir /repo \
        rubiks-img \
        /bin/bash
}

doc_run_nogpu
