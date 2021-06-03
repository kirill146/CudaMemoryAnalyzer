#include "pch.h"
#include "Solver.h"
#include "State.h"

std::optional<z3::model> getErrorModel(State const* state, z3::expr const& rule)  {
    z3::solver s(*state->z3_ctx);

    for (auto const& predicate : state->getPredicates()) {
        s.add(*predicate);
    }

    s.add(rule);

    //std::cout << s.to_smt2() << std::endl;
    //std::cout << "========================\n";
    if (s.check() == z3::sat) {
        //std::cout << s.to_smt2() << std::endl;
        //std::cout << "========================\n";
        //std::cout << s.get_model().to_string() << std::endl;
        return z3::model(s.get_model());
    }
    return std::nullopt;
}

bool isReachable(State* state) {
    z3::solver s(*state->z3_ctx);

    for (auto const& predicate : state->getPredicates()) {
        s.add(*predicate);
    }

    return s.check() == z3::sat;
}

void printState(State* state) {
    z3::solver s(*state->z3_ctx);

    for (auto const& predicate : state->getPredicates()) {
        s.add(*predicate);
    }

    std::cout << s.to_smt2() << std::endl;
    std::cout << "------------------------------------------\n";
}