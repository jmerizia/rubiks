#ifndef _HELPERS_
#define _HELPERS_

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include "helpers.hpp"

/*
 * Ordered list of Rubiks's Cube actions.
 */
std::vector<std::string>
rubiks_action_names = {
    "F",  "R",  "L",  "D",  "U",  "B",  // order matters
    "F'", "R'", "L'", "D'", "U'", "B'"  // order matters
};

template <class U, class V>
std::string
to_string(std::map<U, V> m) {
    std::string s ("test");
    return s;
}

static
std::map<int, int>
read_one_rubiks_action(std::ifstream& f) {
    std::map<int, int> m;
    // fill with defaults:
    for (int i = 1; i <= 54; i++) {
        m[i] = i;
    }
    // fill with information from file:
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            int a, b;
            f >> a >> b;
            m[a] = b;
        }
    }
    return m;
}

static
std::map<int, int>
invert_rubiks_action(std::map<int, int>& m) {
    std::map<int, int> m_inv;
    for (int i = 1; i <= 54; i++) {
        m_inv[m[i]] = i;
    }
    return m_inv;
}


std::map<std::string, Action>
read_rubiks_actions() {
    std::ifstream f;
    f.open("rubiks_actions.in");
    std::map<std::string, Action> actions;
    for (std::string action : {"F", "R", "L", "D", "U", "B"}) {
        std::map<int, int> m = read_one_rubiks_action(f);
        std::map<int, int> m_inv = invert_rubiks_action(m);
        actions[action] = Action(m);
        actions[action + "'"] = Action(m_inv);
    }
    return actions;
}


void
print_state(State& s) {
    char st[1000] =
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n"
        "\n"
        "%2d  %2d  %2d     %2d  %2d  %2d    %2d  %2d  %2d\n"
        "%2d  %2d  %2d     %2d  %2d  %2d    %2d  %2d  %2d\n"
        "%2d  %2d  %2d     %2d  %2d  %2d    %2d  %2d  %2d\n"
        "\n"
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n"
        "\n"
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n"
        "               %2d  %2d  %2d\n\n";
    auto m = s.st;
    printf(st,
                          m[1],  m[2],  m[3],
                          m[4],  m[5],  m[6],
                          m[7],  m[8],  m[9],

    m[46], m[47], m[48],  m[10], m[11], m[12],   m[37], m[38], m[39],
    m[49], m[50], m[51],  m[13], m[14], m[15],   m[40], m[41], m[42],
    m[52], m[53], m[54],  m[16], m[17], m[18],   m[43], m[44], m[45],

                          m[19], m[20], m[21],
                          m[22], m[23], m[24],
                          m[25], m[26], m[27],

                          m[28], m[29], m[30],
                          m[31], m[32], m[33],
                          m[34], m[35], m[36]
          );
}


#endif
