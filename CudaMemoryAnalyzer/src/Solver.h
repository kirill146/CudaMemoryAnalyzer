#pragma once
#include <z3++.h>
#include <optional>

class State;

std::optional<z3::model> getErrorModel(State const* state, z3::expr const& rule);
bool isReachable(State* state);
void printState(State* state);