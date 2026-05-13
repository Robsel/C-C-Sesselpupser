#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <chrono>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Matrix;

double distance(const Vec &a, const Vec &b) // squared Euclidean distance between two vectors
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

double col_distance(const Matrix &weights, int i, int j) // squared Euclidean distance between two neurons
{
    double sum = 0.0;
    for (size_t d = 0; d < weights.size(); d++)
    {
        double diff = weights[d][i] - weights[d][j];
        sum += diff * diff;
    }
    return sum;
}

double col_distance_vec(const Matrix &weights, int i, const Vec &x) // squared Euclidean distance between a neuron and a data point
{
    double sum = 0.0;
    for (size_t d = 0; d < weights.size(); d++)
    {
        double diff = weights[d][i] - x[d];
        sum += diff * diff;
    }
    return sum;
}

Matrix initialize_diagonal(int num_neurons, const Matrix &data) // Initializes weights along the diagonal of the data space
{
    int dim = data[0].size();
    Vec min_vals(dim, 1e9), max_vals(dim, -1e9);
    for (const auto &row : data) // Find min and max for each dimension
        for (int i = 0; i < dim; i++)
        {
            min_vals[i] = min(min_vals[i], row[i]);
            max_vals[i] = max(max_vals[i], row[i]);
        }
    Matrix weights(dim, Vec(num_neurons));
    for (int n = 0; n < num_neurons; n++) // Place neurons at the midpoint along the diagonal in every dimension
        for (int d = 0; d < dim; d++)
            weights[d][n] = min_vals[d] + (max_vals[d] - min_vals[d]) * n / (num_neurons - 1);
    return weights;
}

Matrix initialize_onepoint(int num_neurons, const Matrix &data) // Initializes all weights to the same point, the midpoint of each dimension in the data set
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

Matrix initialize_random(int num_neurons, const Matrix &data) // Initializes weights randomly within the bounds of the data set for each dimension
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

    int num_neurons, epochs, update_neighbors_epoch, calculate_k_epoch, k_neighbors;
    double learning_rate, influence;

    SOM(const Matrix &data, int num_neurons, int epochs = 100,
        double learning_rate = 0.3, double influence = 0.1,
        int update_neighbors_epoch = 4, int calculate_k_epoch = 6,
        int k_neighbors = 4, InitMode init_mode = DIAGONAL)
        : data(data), num_neurons(num_neurons), epochs(epochs),
          learning_rate(learning_rate), influence(influence),
          update_neighbors_epoch(update_neighbors_epoch),
          calculate_k_epoch(calculate_k_epoch), k_neighbors(k_neighbors)
    {
        if (init_mode == DIAGONAL)
            weights = initialize_diagonal(num_neurons, data);
        else if (init_mode == ONEPOINT)
            weights = initialize_onepoint(num_neurons, data);
        else
            weights = initialize_random(num_neurons, data);
        neighbors.resize(num_neurons);
    }

    void calculate_neighbors() // computes the k nearest neighbors for each neuron based on location of other neurons in the weight space
    {
        for (int i = 0; i < num_neurons; i++)
        {
            vector<pair<double, int>> dist_list;
            for (int j = 0; j < num_neurons; j++)
                dist_list.push_back({col_distance(weights, i, j), j});
            sort(dist_list.begin(), dist_list.end());
            neighbors[i].clear();
            for (int k = 0; k < k_neighbors; k++)
                neighbors[i].push_back(dist_list[k].second);
        }
    }

    void update_neighborhood(int idx, const Vec &x) // updates the weights for the neighboring neurons of the neuron closest to the current point
    {
        int dim = weights.size();
        for (int n : neighbors[idx])
            for (int d = 0; d < dim; d++)
                weights[d][n] += influence * learning_rate * (x[d] - weights[d][n]);
    }

    int closest_neuron(const Vec &x) // finds the closest neuron to a given data point by computing the distance from the point to each neuron and returning the index of the neuron with the smallest distance
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

    vector<int> train() // main training loop iterating over the specified number of epochs
    {
        int dim = weights.size();

        calculate_neighbors(); // define initial neighbors before training starts
        random_device rd;
        mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            shuffle(data.begin(), data.end(), gen); // Shuffle data points each epoch for more accurate approximation of the data distribution
            bool do_neighbor_update = (epoch >= update_neighbors_epoch &&
                                       epoch % update_neighbors_epoch == 0);
            bool do_calc_neighbors = (epoch >= calculate_k_epoch &&
                                      epoch % calculate_k_epoch == 0);

            for (const auto &x : data) // loop over every data point
            {
                int idx = closest_neuron(x);  // Find the closest neuron to the current data point
                for (int d = 0; d < dim; d++) // Update the weights of the closest neuron towards the current data point
                    weights[d][idx] += learning_rate * (x[d] - weights[d][idx]);
                if (do_neighbor_update)
                {
                    update_neighborhood(idx, x);
                }
            }
            if (do_calc_neighbors)
            {
                calculate_neighbors();
            }
        }
        vector<int> clusters;
        for (const auto &x : data) // After training, assign each data point to the cluster of its closest neuron
            clusters.push_back(closest_neuron(x));
        return clusters;
    }
};

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
            row.push_back(val);
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
    if (argc < 2)
    {
        cerr << "Error: No data file provided.\n";
        cerr << "Usage: ./a.out datafile1 datafile2 ...]\n";
        return 404;
    }

    int num_files = argc - 1;
    vector<string> filenames(num_files);
    for (int i = 0; i < num_files; i++)
        filenames[i] = argv[i + 1];

    // --- Helper lambdas ---
    auto read_int = [&](const string &prompt, int dv) -> int
    {
        cout << prompt << " [" << dv << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return dv;
        stringstream ss(line);
        int v;
        return (ss >> v) ? v : dv;
    };
    auto read_double = [&](const string &prompt, double dv) -> double
    {
        cout << prompt << " [" << dv << "]: ";
        string line;
        getline(cin, line);
        if (line.empty())
            return dv;
        stringstream ss(line);
        double v;
        return (ss >> v) ? v : dv;
    };

    // --- SOM parameter struct ---
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

    // --- Load all datasets ---
    vector<Matrix> datasets(num_files);
    for (int i = 0; i < num_files; i++)
    {
        datasets[i] = load_data(filenames[i]);
    }

    // --- Ask parameters for each SOM in sequence ---
    vector<SOMParams> all_params(num_files);

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

        int init_in = read_int("  Init mode (0=diagonal, 1=onepoint, 2=random)", def_init);
        if (init_in == 1)
            all_params[i].init_mode = SOM::ONEPOINT;
        else if (init_in == 2)
            all_params[i].init_mode = SOM::RANDOM;
        else
            all_params[i].init_mode = SOM::DIAGONAL;

        // carry forward as defaults for the next SOM
        def_neurons = all_params[i].num_neurons;
        def_epochs = all_params[i].epochs;
        def_lr = all_params[i].learning_rate;
        def_influence = all_params[i].influence;
        def_upd_nb = all_params[i].update_neighbors_epoch;
        def_calc_k = all_params[i].calculate_k_epoch;
        def_k_nb = all_params[i].k_neighbors;
        def_init = init_in;
    }

    // --- Train each SOM ---
    vector<vector<int>> all_clusters(num_files);
    double start_time = chrono::duration<double>(chrono::high_resolution_clock::now().time_since_epoch()).count();
    for (int i = 0; i < num_files; i++)
    {
        const SOMParams &p = all_params[i];
        SOM som(datasets[i],
                p.num_neurons, p.epochs, p.learning_rate, p.influence,
                p.update_neighbors_epoch, p.calculate_k_epoch,
                p.k_neighbors, p.init_mode);

        all_clusters[i] = som.train();
    }
    double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now().time_since_epoch()).count() - start_time;

    // --- Cluster counts ---
    for (int i = 0; i < num_files; i++)
    {
        cout << "\nSom n°" << (i + 1) << " Clusters:\n";
        map<int, int> count;
        for (int c : all_clusters[i])
            count[c]++;
        for (const auto &kv : count)
            cout << "  Neuron " << kv.first << ": " << kv.second << " point(s)\n";
    }

    cout << "Total training time for all SOMs: " << elapsed << " seconds\n";
    return 0;
}