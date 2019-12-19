#ifndef _MAIN_
#define _MAIN_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <dlib/dnn.h>

#include "helpers.hpp"

using namespace std;
using namespace dlib;

/*
 * Given an initial state, and a final state,
 * return a path of Actions to that state.
 */
std::vector<Action>
search(
    State start,
    State target
) {
    std::vector<Action> v;
    return v;
}

double dist(int r, int c) {
    return sqrt((double)r*r+c*c);
}

int
main(void) {

    std::map<std::string, Action> actions = read_rubiks_actions();

    //std::cout << actions["F"].size() << std::endl;

    using input_type = matrix<float, 2, 1>;

    std::vector<input_type> data;
    std::vector<float> labels;
    for (int r = 0; r <= 20; r++) {
        for (int c = 0; c <= 20; c++) {
            input_type point;
            point = ((double)r) / 20.0,
                    ((double)c) / 20.0;
            data.push_back(point);
            labels.push_back(dist(r, c) / 30.0);
        }
    }

    using net_type = loss_mean_squared<
                     relu<fc<1,
                     relu<fc<10,
                     input<matrix<float, 2, 1
                     >>>>>>>;

    net_type net;


    //trainer.set_synchronization_file("test_sync", std::chrono::seconds(20));

    input_type t;
    for (int i = 0; i < 100; i++) {

        // create trainer and train:
        dnn_trainer<net_type> trainer(net);
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.01);
        trainer.set_max_num_epochs(10);
        //trainer.be_verbose();
        trainer.train(data, labels);

        // see the error:
        double total_error = 0.0;
        int total_tests = 0;
        for (int r = 0; r <= 20; r++) {
            for (int c = 0; c <= 20; c++) {
                t = ((double)r) / 20.0,
                    ((double)c) / 20.0;
                double diff = net(t) * 30 - dist(r, c);
                total_error += diff*diff;
                total_tests++;
            }
        }
        cout << total_error / total_tests << endl;
    }

    //net.clean();
    //serialize("test_network.dat") << net;


    //State s;
    //State sp = s.apply(actions["B"]);

    //print_state(s);
    //print_state(sp);

    return 0;
}

#endif
