#ifndef _MAIN_
#define _MAIN_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <dlib/bigint.h>

#include "helpers.hpp"

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


int
main(void) {

    std::map<std::string, Action> actions = read_rubiks_actions();

    //std::cout << actions["F"].size() << std::endl;

    State s;
    State sp = s.apply(actions["B"]);

    print_state(s);
    print_state(sp);

    return 0;
}

#endif
