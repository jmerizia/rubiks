#ifndef _HELPERS_H_
#define _HELPERS_H_

#define CONTAINS(s, x) ((s).find(x) != (s).end())

#include <iostream>
#include <string>
#include <map>

class Action;
class State;

std::map<std::string, Action>
read_rubiks_actions();

template <class U, class V>
std::string
to_string(std::map<U, V>);

void
print_state(State& s);

/*
 * A mapping that takes a game state
 * and produces another.
 */
class Action {
    public:
        std::map<int /* slot idx */, int /* slot idx */> perm;
        Action() {}
        explicit Action(std::map<int, int> p): perm(p) { }
        int size() {
            return this->perm.size();
        }
};

/*
 * Stores a state of the rubiks cube.
 */
class State {
    public:
        std::map<int /* slot idx */, int /* sticker idx */> st;

        State() {
            for (int i = 1; i <= 54; i++) {
                this->st[i] = i;
            }
        }

        /*
         * Create a State from an integer map.
         */
        explicit State(std::map<int, int> s): st(s) { }

        /*
         * Apply the given action to this state,
         * and return a new state.
         */
        State apply(Action a) {
            std::map<int, int> m;
            for (int i = 1; i <= 54; i++) {
                m[a.perm[i]] = this->st[i];
            }
            return State(m);
        }
};


#endif
