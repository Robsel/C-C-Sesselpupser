#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <map>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Matrix;

// Euclidean distance
double distance(const Vec &a, const Vec &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

// --- Initialization methods ---

Matrix initialize_diagonal(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);
    for (const auto &row : data)
    {
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    }
    Matrix weights(num_neurons, Vec(dim));

    for (int i = 0; i < num_neurons; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            weights[i][d] = min_vals[d] +
                            (max_vals[d] - min_vals[d]) * i / (num_neurons - 1);
        }
    }
    return weights;
}

Matrix initialize_onepoint(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);

    for (const auto &row : data)
    {
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    }

    Vec mid(dim);
    for (int i = 0; i < dim; i++)
    {
        mid[i] = (min_vals[i] + max_vals[i]) / 2.0;
    }

    Matrix weights(num_neurons, mid); // replicate midpoint
    return weights;
}

Matrix initialize_random(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);

    for (const auto &row : data)
    {
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    }

    random_device rd;
    mt19937 gen(rd());
    Matrix weights(num_neurons, Vec(dim));

    for (int i = 0; i < num_neurons; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            uniform_real_distribution<> dis(min_vals[d], max_vals[d]);
            weights[i][d] = dis(gen);
        }
    }
    return weights;
}

// --- SOM Class ---

class SOM
{
public:
    enum InitMode
    {
        DIAGONAL,
        ONEPOINT,
        RANDOM
    };

    Matrix data;
    Matrix weights;
    vector<vector<int>> neighbors;

    int num_neurons;
    int epochs;
    double learning_rate;
    double influence;
    int update_neighbors_epoch;
    int calculate_k_epoch;
    int k_neighbors;

    SOM(const Matrix &data,
        int num_neurons,
        int epochs = 100,
        double learning_rate = 0.3,
        double influence = 0.1,
        int update_neighbors_epoch = 4,
        int calculate_k_epoch = 6,
        int k_neighbors = 4,
        InitMode init_mode = DIAGONAL)
        : data(data),
          num_neurons(num_neurons),
          epochs(epochs),
          learning_rate(learning_rate),
          influence(influence),
          update_neighbors_epoch(update_neighbors_epoch),
          calculate_k_epoch(calculate_k_epoch),
          k_neighbors(k_neighbors)
    {
        if (init_mode == DIAGONAL)
            weights = initialize_diagonal(num_neurons, data);
        else if (init_mode == ONEPOINT)
            weights = initialize_onepoint(num_neurons, data);
        else
            weights = initialize_random(num_neurons, data);

        neighbors.resize(num_neurons);
    }

    void calculate_neighbors()
    {

        for (int i = 0; i < num_neurons; i++)
        {
            vector<pair<double, int>> dist_list;

            for (int j = 0; j < num_neurons; j++)
            {
                if (i == j)
                    continue;
                dist_list.push_back({distance(weights[i], weights[j]), j});
            }

            sort(dist_list.begin(), dist_list.end());

            neighbors[i].clear();
            for (int k = 0; k < k_neighbors && k < dist_list.size(); k++)
            {
                neighbors[i].push_back(dist_list[k].second);
            }
        }
    }

    void update_neighborhood(int idx, const Vec &x)
    {

        for (int n : neighbors[idx])
        {
            for (size_t d = 0; d < weights[n].size(); d++)
            {
                weights[n][d] += influence * learning_rate * (x[d] - weights[n][d]);
            }
        }
    }

    int closest_neuron(const Vec &x) const
    {
        double best_dist = 1e18;
        int best_idx = 0;

        for (int i = 0; i < num_neurons; i++)
        {
            double d = distance(weights[i], x);
            if (d < best_dist)
            {
                best_dist = d;
                best_idx = i;
            }
        }
        return best_idx;
    }

    vector<int> train()
    {
        calculate_neighbors();

        random_device rd;
        mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            shuffle(data.begin(), data.end(), gen);

            for (const auto &x : data)
            {
                int idx = closest_neuron(x);
#pragma omp parallel for
                for (size_t d = 0; d < weights[idx].size(); d++)
                {
                    weights[idx][d] += learning_rate * (x[d] - weights[idx][d]);
                }

                if (epoch >= update_neighbors_epoch &&
                    epoch % update_neighbors_epoch == 0)
                {
                    update_neighborhood(idx, x);
                }
            }

            if (epoch >= calculate_k_epoch &&
                epoch % calculate_k_epoch == 0)
            {
                calculate_neighbors();
            }
        }

        vector<int> clusters(data.size());

        for (int idx = 0; idx < data.size(); idx++)
        {
            clusters[idx] = closest_neuron(data[idx]);
        }

        return clusters;
    }
};

//------------Main function and helper to load data from file--------------

// --- Helper: load data from file ---
Matrix load_data(const string &filename)
{
    ifstream file(filename);
    Matrix data;

    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        Vec row;
        double val;

        while (ss >> val)
        {
            row.push_back(val);
        }

        if (!row.empty())
            data.push_back(row);
    }

    if (data.empty())
    {
        cerr << "Error: No valid data found in file." << endl;
        exit(1);
    }

    return data;
}

// --- MAIN ---
int main(int argc, char *argv[])
{

    // Check if data file was provided
    if (argc < 2)
    {
        std::cerr << "Error: No data file provided.\n";
        std::cerr << "Usage: ./a.out <datafile>\n";
        return 404;
    }

    string filename = argv[1];
    Matrix data = load_data(filename);

    // Parameters and defaults
    int num_neurons, epochs, update_neighbors_epoch, calculate_k_epoch, k_neighbors;
    double learning_rate, influence;
    int init_mode_input;
    int default_num_neurons = 10;
    int default_epochs = 100;
    double default_learning_rate = 0.3;
    double default_influence = 0.1;
    int default_update_neighbors_epoch = 4;
    int default_calculate_k_epoch = 6;
    int default_k_neighbors = 4;
    int default_init_mode_input = 0;

    auto read_int = [&](const string &prompt, int default_val)
    {
        cout << prompt << " [" << default_val << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return default_val;
        stringstream ss(line);
        int value;
        if (ss >> value)
            return value;
        return default_val;
    };

    auto read_double = [&](const string &prompt, double default_val)
    {
        cout << prompt << " [" << default_val << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return default_val;
        stringstream ss(line);
        double value;
        if (ss >> value)
            return value;
        return default_val;
    };

    num_neurons = read_int("Enter number of neurons", default_num_neurons);
    epochs = read_int("Enter number of epochs", default_epochs);
    learning_rate = read_double("Enter learning rate", default_learning_rate);
    influence = read_double("Enter influence", default_influence);
    update_neighbors_epoch = read_int("Enter update_neighbors_epoch", default_update_neighbors_epoch);
    calculate_k_epoch = read_int("Enter calculate_k_epoch", default_calculate_k_epoch);
    k_neighbors = read_int("Enter k_neighbors", default_k_neighbors);
    init_mode_input = read_int("Init mode (0 = diagonal, 1 = onepoint, 2 = random)", default_init_mode_input);

    SOM::InitMode init_mode;
    if (init_mode_input == 0)
        init_mode = SOM::DIAGONAL;
    else if (init_mode_input == 1)
        init_mode = SOM::ONEPOINT;
    else
        init_mode = SOM::RANDOM;

    // Create SOM
    SOM som(data,
            num_neurons,
            epochs,
            learning_rate,
            influence,
            update_neighbors_epoch,
            calculate_k_epoch,
            k_neighbors,
            init_mode);

    double start_time = omp_get_wtime();
    // Train
    vector<int> clusters = som.train();

    double end_time = omp_get_wtime();
    cout << "Training completed in " << (end_time - start_time) << " seconds." << endl;

    std::map<int, int> count;

    // Count occurrences
    for (int num : clusters)
    {
        count[num]++;
    }

    // Print results (automatically sorted)
    for (const auto &pair : count)
    {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}