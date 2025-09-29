#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <unordered_map>

using namespace std;

// Represents a city with ID and coordinates
struct City {
    int id;
    double x, y;
};

// Calculate Euclidean distance (straight line) between two cities
double distanceBetween(const City& a, const City& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Compute the total distance of a route
double routeDistance(const vector<int>& route,
                     const vector<City>& cities,
                     const unordered_map<int, int>& idToIndex) 
{
    double dist = 0.0;
    for (size_t i = 1; i < route.size(); ++i) {
        const City& from = cities[idToIndex.at(route[i - 1])];
        const City& to   = cities[idToIndex.at(route[i])];
        dist += distanceBetween(from, to);
    }
    return dist;
}

// Try to improve a route using the 2-opt algorithm
// It swaps two edges if that makes the total route shorter
int twoOpt(vector<int>& route,
           const vector<City>& cities,
           const unordered_map<int, int>& idToIndex,
           double& globalBest) 
{
    bool improved = true;
    int improvements = 0;

    while (improved) {
        improved = false;

        for (size_t i = 1; i < route.size() - 2; ++i) {
            for (size_t k = i + 1; k < route.size() - 1; ++k) {
                int a = route[i - 1], b = route[i];
                int c = route[k], d = route[k + 1];

                // Difference in distance if we swap
                double delta = (distanceBetween(cities[idToIndex.at(a)], cities[idToIndex.at(c)]) +
                                distanceBetween(cities[idToIndex.at(b)], cities[idToIndex.at(d)]))
                             - (distanceBetween(cities[idToIndex.at(a)], cities[idToIndex.at(b)]) +
                                distanceBetween(cities[idToIndex.at(c)], cities[idToIndex.at(d)]));

                // If the swap improves the route, perform it
                if (delta < -1e-6) {
                    reverse(route.begin() + i, route.begin() + k + 1);
                    improved = true;
                    improvements++;

                    double currentDist = routeDistance(route, cities, idToIndex);
                    if (currentDist < globalBest) {
                        globalBest = currentDist;
                    }
                }
            }
        }
    }
    return improvements;
}

int main() {
    srand(static_cast<unsigned>(time(0))); // random seed

    // Open the TSP file
    ifstream in("berlin52.tsp");
    if (!in) {
        cerr << "Could not open file \"berlin52.tsp\".\n";
        return 1;
    }

    string line;
    bool foundSection = false;

    // Skip until NODE_COORD_SECTION
    while (getline(in, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            foundSection = true;
            break;
        }
    }
    if (!foundSection) {
        cerr << "NODE_COORD_SECTION not found in file.\n";
        return 1;
    }
    
    // Read city data
    vector<City> cities;
    while (getline(in, line)) {
        if (line.find("EOF") != string::npos) break;
        if (line.empty()) continue;

        istringstream iss(line);
        City c;
        if (iss >> c.id >> c.x >> c.y) {
            cities.push_back(c);
        }
    }
    if (cities.empty()) {
        cerr << "No cities were read from the file.\n";
        return 1;
    }

    // Map city ID -> index in vector
    unordered_map<int, int> idToIndex;
    for (size_t i = 0; i < cities.size(); ++i) {
        idToIndex[cities[i].id] = static_cast<int>(i);
    }

    // Start at city with ID = 1
    int startID = 1;
    if (idToIndex.find(startID) == idToIndex.end()) {
        cerr << "Start city with ID 1 not found.\n";
        return 1;
    }

    vector<int> bestRoute;
    double bestDistance = numeric_limits<double>::infinity();
    double globalBest = numeric_limits<double>::infinity();

    // Try 2000 random routes
    for (int attempt = 0; attempt < 2000; ++attempt) {
        // Build a route: start -> all cities -> back to start
        vector<int> route;
        route.push_back(startID);
        for (size_t i = 1; i <= cities.size(); ++i) {
            if (static_cast<int>(i) != startID) {
                route.push_back(i);
            }
        }
        random_shuffle(route.begin() + 1, route.end()); 
        route.push_back(startID);

        // Improve with 2-opt
        twoOpt(route, cities, idToIndex, globalBest);

        // Calculate distance
        double dist = routeDistance(route, cities, idToIndex);

        if (dist < bestDistance) {
            bestDistance = dist;
            bestRoute = route;
        }
    }

    // Print results
    cout << fixed << setprecision(3);
    cout << "Best route (city IDs): ";
    for (int id : bestRoute) cout << id << " ";
    cout << "\nTotal distance: " << bestDistance << "\n";

    if (bestDistance <= 8000.0) 
        cout << "Route is within constraint (<= 8000).\n";
    else 
        cout << "Route is above constraint (> 8000).\n";

    return 0;
}
