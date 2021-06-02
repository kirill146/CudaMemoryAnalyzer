#include <iostream>
#include "json.hpp"
#include <fstream>

int main() {
	std::ifstream fin("vectorAdd.json");
	if (!fin.is_open()) {
		std::cout << "Cannot open file" << std::endl;
	}
	nlohmann::json params;
	try {
		fin >> params;
		std::cout << params.at("file") << std::endl;
		std::cout << params.at("kernelName") << std::endl;
		for (auto const& dir : params.at("additionalIncludeDirs")) {
			std::cout << dir << std::endl;
		}
		
	} catch (nlohmann::json::parse_error& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	return 0;
}