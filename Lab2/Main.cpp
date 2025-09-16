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
#include <filesystem>   

using namespace std;
// Assign Id and coordinates to every citys
struct City {
    int id;
    double x, y;
};
// With pythagoras the distance is calculated between two citys
double euclid(const City& a, const City& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Calculates the total distance of all routes
double total_distance(const vector<int>& route, const vector<City>& cities, const unordered_map<int, int>& id2idx) {
    double dist = 0.0;
    for (size_t i = 1; i < route.size(); ++i) {
        dist += euclid(cities[id2idx.at(route[i - 1])], cities[id2idx.at(route[i])]);
    }
    return dist;
}

// Compares routes and switches out if another route is shorter than the previously stored
int two_opt(vector<int>& route, const vector<City>& cities, const unordered_map<int, int>& id2idx, double &globalBest) {
    bool improved = true;
    long long iteration = 0;       // totalt antal iterationer
    int improvementCount = 0;
    while (improved) {
        improved = false;
        for (size_t i = 1; i < route.size() - 2; ++i) {
            for (size_t k = i + 1; k < route.size() - 1; ++k) {
                int a = route[i - 1], b = route[i];
                int c = route[k], d = route[k + 1];
                double delta = (euclid(cities[id2idx.at(a)], cities[id2idx.at(c)]) +
                    euclid(cities[id2idx.at(b)], cities[id2idx.at(d)]))
                    - (euclid(cities[id2idx.at(a)], cities[id2idx.at(b)]) +
                        euclid(cities[id2idx.at(c)], cities[id2idx.at(d)]));
                if (delta < -1e-6) {
                    reverse(route.begin() + i, route.begin() + k + 1);
                    improved = true;

                    improvementCount++;
                    double currentDist = total_distance(route, cities, id2idx);
                    if (currentDist < globalBest) {
                        globalBest = currentDist;
                        cout << "Global improvement! distance = " << globalBest
                            << " (trial improvement #" << improvementCount
                            << ", iteration " << iteration << ")\n";
                    }
                }
                    
            }
        }
    } 
    
    return improvementCount;
}

int main() {



    srand(static_cast<unsigned>(time(0))); // random seed
    // Reads the TSP file
    ifstream in("berlin52.tsp");
    if (!in) {
        cerr << "Kunde inte öppna filen \"" << "berlin52.tsp" << "\"\n";
        return 1;
    }

    string line;
    // Jumps to NODE_COORD_SECTION
    bool found = false;
    while (getline(in, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            found = true;
            break;
        }
    }
    if (!found) {
        cerr << "Hittade inte NODE_COORD_SECTION i filen.\n";
        return 1;
    }
    
    // Reads Cities until end
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
        cerr << "Inga städer lästa från filen.\n";
        return 1;
    }

    // Makes a table so cities can be connected with its ID
    unordered_map<int, int> id2idx;
    int max_id = 0;
    for (size_t i = 0; i < cities.size(); ++i) {
        id2idx[cities[i].id] = static_cast<int>(i);
        if (cities[i].id > max_id) max_id = cities[i].id;
    }


    int startID = 1;
    if (id2idx.find(startID) == id2idx.end()) {
        cerr << "Startstad med ID 1 hittades inte.\n";
        return 1;
    }

    vector<int> bestRoute;
    double bestDist = numeric_limits<double>::infinity();
    double globalBest = numeric_limits<double>::infinity();
    int totalImprovements = 0;
    // Runs 2000 random tries
    for (int trial = 0; trial < 2000; ++trial) {
        // Creates a route 1->2..n-1->1
        vector<int> route;
        route.push_back(startID);
        for (size_t i = 1; i <= cities.size(); ++i) {
            if (static_cast<int>(i) != startID) route.push_back(i);
        }
        random_shuffle(route.begin() + 1, route.end()); // shuffle bara mellan första och sista
        route.push_back(startID);

        int improvementsThisRoute = two_opt(route, cities, id2idx, globalBest);
        totalImprovements += improvementsThisRoute;

        double dist = total_distance(route, cities, id2idx);
        cout << "Generation " << trial + 1 << " best distance = " << bestDist << "\n";

        if (dist < bestDist) {
            bestDist = dist;
            bestRoute = route;
        }
    }

    // Prints the results
    cout << fixed << setprecision(3);
    cout << "Route (IDs): ";
    for (int id : bestRoute) cout << id << " ";
    cout << "\nTotal distance: " << bestDist << "\n";
    //cout << "Totalt antal förbättringar över alla starter: " << totalImprovements << "\n";
    if (bestDist <= 8000.0) cout << "Within constraint (<= 8000)\n";
    else cout << "Over constraint (> 8000)\n";

    return 0;
}