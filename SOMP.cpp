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

// Euclidean distance between two full vectors
double distance(const Vec &a, const Vec &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

// Euclidean distance between two neuron columns in a [dim][num_neurons] weight matrix
double col_distance(const Matrix &weights, int i, int j)
{
    double sum = 0.0;
    for (size_t d = 0; d < weights.size(); d++)
    {
        double diff = weights[d][i] - weights[d][j];
        sum += diff * diff;
    }
    return sum;
}

// Distance between a data point x and neuron column i
double col_distance_vec(const Matrix &weights, int i, const Vec &x)
{
    double sum = 0.0;
    for (size_t d = 0; d < weights.size(); d++)
    {
        double diff = weights[d][i] - x[d];
        sum += diff * diff;
    }
    return sum;
}
Matrix initialize_diagonal(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);
    for (const auto &row : data)
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    Matrix weights(dim, Vec(num_neurons));
    for (int n = 0; n < num_neurons; n++)
        for (int d = 0; d < dim; d++)
            weights[d][n] = min_vals[d] + (max_vals[d] - min_vals[d]) * n / (num_neurons - 1);
    return weights;
}

Matrix initialize_onepoint(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);
    for (const auto &row : data)
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    Matrix weights(dim, Vec(num_neurons));
    for (int d = 0; d < dim; d++)
    {
        double mid = (min_vals[d] + max_vals[d]) / 2.0;
        for (int n = 0; n < num_neurons; n++)
            weights[d][n] = mid;
    }
    return weights;
}

Matrix initialize_random(int num_neurons, const Matrix &data)
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);
    for (const auto &row : data)
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    random_device rd;
    mt19937 gen(rd());
    Matrix weights(dim, Vec(num_neurons));
    for (int d = 0; d < dim; d++)
    {
        uniform_real_distribution<> dis(min_vals[d], max_vals[d]);
        for (int n = 0; n < num_neurons; n++)
            weights[d][n] = dis(gen);
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
            dist_list.reserve(num_neurons - 1);
            for (int j = 0; j < num_neurons; j++)
            {
                if (i == j)
                    continue;
                dist_list.push_back({col_distance(weights, i, j), j});
            }

            sort(dist_list.begin(), dist_list.end());

            neighbors[i].clear();
            for (int k = 0; k < k_neighbors && k < (int)dist_list.size(); k++)
            {
                neighbors[i].push_back(dist_list[k].second);
            }
        }
    }

    void update_neighborhood(int idx, const Vec &x)
    {
        int dim = weights.size();

        for (int n : neighbors[idx])
        {
            for (int d = 0; d < dim; d++)
            {
                weights[d][n] += influence * learning_rate * (x[d] - weights[d][n]);
            }
        }
    }

    int closest_neuron(const Vec &x, const Matrix &weights) const
    {
        double best_dist = 1e18;
        int best_idx = 0;

        for (int i = 0; i < num_neurons; i++)
        {
            double d = col_distance_vec(weights, i, x);
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
        int dim = weights.size();
        calculate_neighbors();

        random_device rd;
        mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            shuffle(data.begin(), data.end(), gen);
            for (const auto &x : data)
            {
                int idx = closest_neuron(x, weights);

                for (int d = 0; d < dim; d++)
                {
                    weights[d][idx] += learning_rate * (x[d] - weights[d][idx]);
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
        for (int idx = 0; idx < (int)data.size(); idx++)
        {
            clusters[idx] = closest_neuron(data[idx], weights);
        }

        return clusters;
    }
};

//------------Helper to load data from file--------------

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
        cerr << "Error: No valid data found in file: " << filename << endl;
        exit(1);
    }

    return data;
}

// --- MAIN ---
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Error: No data file provided.\n";
        cerr << "Usage: ./a.out <datafile1> [datafile2 ...]\n";
        return 404;
    }

    // Collect all input filenames (one or more)
    int num_files = argc - 1;
    vector<string> filenames(num_files);
    for (int i = 0; i < num_files; i++)
        filenames[i] = argv[i + 1];

    // --- Per-SOM parameter struct ---
    struct SOMParams
    {
        int num_neurons;
        int epochs;
        double learning_rate;
        double influence;
        int update_neighbors_epoch;
        int calculate_k_epoch;
        int k_neighbors;
        SOM::InitMode init_mode;
    };

    // --- Helper lambdas for interactive input ---
    auto read_int = [&](const string &prompt, int default_val) -> int
    {
        cout << prompt << " [" << default_val << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return default_val;
        stringstream ss(line);
        int value;
        return (ss >> value) ? value : default_val;
    };

    auto read_double = [&](const string &prompt, double default_val) -> double
    {
        cout << prompt << " [" << default_val << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return default_val;
        stringstream ss(line);
        double value;
        return (ss >> value) ? value : default_val;
    };

    auto read_init_mode = [&](int default_val) -> SOM::InitMode
    {
        int v = read_int("Init mode (0=diagonal, 1=onepoint, 2=random)", default_val);
        if (v == 1)
            return SOM::ONEPOINT;
        if (v == 2)
            return SOM::RANDOM;
        return SOM::DIAGONAL;
    };
    // --- Pre-load all datasets ---
    vector<Matrix> datasets(num_files);
    for (int i = 0; i < num_files; i++)
        datasets[i] = load_data(filenames[i]);

    // --- Ask parameters for each SOM in sequence ---
    vector<SOMParams> all_params(num_files);

    // Defaults
    int def_neurons = 10;
    int def_epochs = 100;
    double def_lr = 0.3;
    double def_influence = 0.1;
    int def_upd_nb = 4;
    int def_calc_k = 6;
    int def_k_nb = 4;
    int def_init = 0;

    for (int i = 0; i < num_files; i++)
    {
        cout << "\n--- Parameters for Som n°" << (i + 1)
             << " [" << filenames[i] << "] ---\n";

        all_params[i].num_neurons = read_int("  Number of neurons", def_neurons);
        all_params[i].epochs = read_int("  Number of epochs", def_epochs);
        all_params[i].learning_rate = read_double("  Learning rate", def_lr);
        all_params[i].influence = read_double("  Influence", def_influence);
        all_params[i].update_neighbors_epoch = read_int("  update_neighbors_epoch", def_upd_nb);
        all_params[i].calculate_k_epoch = read_int("  calculate_k_epoch", def_calc_k);
        all_params[i].k_neighbors = read_int("  k_neighbors", def_k_nb);
        all_params[i].init_mode = read_init_mode(def_init);

        def_neurons = all_params[i].num_neurons;
        def_epochs = all_params[i].epochs;
        def_lr = all_params[i].learning_rate;
        def_influence = all_params[i].influence;
        def_upd_nb = all_params[i].update_neighbors_epoch;
        def_calc_k = all_params[i].calculate_k_epoch;
        def_k_nb = all_params[i].k_neighbors;

        if (all_params[i].init_mode == SOM::ONEPOINT)
            def_init = 1;
        else if (all_params[i].init_mode == SOM::RANDOM)
            def_init = 2;
        else
            def_init = 0;
    }

    // --- Results storage (pre-allocated so threads write to disjoint slots) ---
    vector<vector<int>> all_clusters(num_files);
    vector<double> all_times(num_files, 0.0);

    // --- Parallel training: one thread per file ---
    double start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_files; i++)
    {
        const SOMParams &p = all_params[i];

        SOM som(datasets[i],
                p.num_neurons,
                p.epochs,
                p.learning_rate,
                p.influence,
                p.update_neighbors_epoch,
                p.calculate_k_epoch,
                p.k_neighbors,
                p.init_mode);

        all_clusters[i] = som.train();
    }
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    // --- Print results in file order ---
    for (int i = 0; i < num_files; i++)
    {
        cout << "Som n°" << (i + 1) << " Clusters:  ["
             << filenames[i] << "] \n";

        map<int, int> count;
        for (int c : all_clusters[i])
            count[c]++;

        for (const auto &p : count)
            cout << "  Neuron " << p.first << ": " << p.second << " point(s)\n";

        cout << endl;
    }
    cout << "Total training time for all SOMs: " << total_time << " seconds\n";
    return 0;
}