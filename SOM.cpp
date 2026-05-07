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

// Helper for high-resolution timing
static double now_sec()
{
    return chrono::high_resolution_clock::now().time_since_epoch().count() / 1e9;
}

double distance(const Vec &a, const Vec &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

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

    // Accumulated timing buckets (seconds)
    double t_shuffle = 0, t_closest = 0, t_weight_update = 0,
           t_neighborhood_update = 0, t_calc_neighbors = 0;
    // Call counts
    long long calls_closest = 0, calls_neighborhood = 0, calls_calc_neighbors = 0;

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

    void calculate_neighbors()
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

    void update_neighborhood(int idx, const Vec &x)
    {
        int dim = weights.size();
        for (int n : neighbors[idx])
            for (int d = 0; d < dim; d++)
                weights[d][n] += influence * learning_rate * (x[d] - weights[d][n]);
    }

    // Add these to the SOM class member variables:
    double t_cn_loop = 0, t_cn_dist = 0, t_cn_compare = 0;

    int closest_neuron(const Vec &x) // remove const to allow timing member updates
    {
        double best_dist = 1e18;
        int best_idx = 0;

        double t_dist_local = 0, t_cmp_local = 0;
        double t0, t1;

        for (int i = 0; i < num_neurons; i++)
        {
            t0 = now_sec();
            double d = col_distance_vec(weights, i, x);
            t1 = now_sec();
            t_dist_local += t1 - t0;

            t0 = now_sec();
            if (d < best_dist)
            {
                best_dist = d;
                best_idx = i;
            }
            t1 = now_sec();
            t_cmp_local += t1 - t0;
        }

        t_cn_dist += t_dist_local;
        t_cn_compare += t_cmp_local;

        return best_idx;
    }

    vector<int> train()
    {
        int dim = weights.size();
        double t0, t1;

        // --- Initial neighbor calculation ---
        t0 = now_sec();
        calculate_neighbors();
        t1 = now_sec();
        t_calc_neighbors += t1 - t0;
        calls_calc_neighbors++;
        cout << "[init] calculate_neighbors: " << (t1 - t0) * 1000.0 << " ms\n";

        random_device rd;
        mt19937 gen(rd());

        double epoch_start = now_sec();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double ep0 = now_sec();

            // --- Shuffle ---
            t0 = now_sec();
            shuffle(data.begin(), data.end(), gen);
            t1 = now_sec();
            t_shuffle += t1 - t0;

            bool do_neighbor_update = (epoch >= update_neighbors_epoch &&
                                       epoch % update_neighbors_epoch == 0);
            bool do_calc_neighbors = (epoch >= calculate_k_epoch &&
                                      epoch % calculate_k_epoch == 0);

            double ep_closest = 0, ep_weight = 0, ep_neighborhood = 0;

            for (const auto &x : data)
            {
                // --- closest_neuron ---
                t0 = now_sec();
                int idx = closest_neuron(x);
                t1 = now_sec();
                ep_closest += t1 - t0;
                calls_closest++;

                // --- weight update ---
                t0 = now_sec();
                for (int d = 0; d < dim; d++)
                    weights[d][idx] += learning_rate * (x[d] - weights[d][idx]);
                t1 = now_sec();
                ep_weight += t1 - t0;

                // --- neighborhood update ---
                if (do_neighbor_update)
                {
                    t0 = now_sec();
                    update_neighborhood(idx, x);
                    t1 = now_sec();
                    ep_neighborhood += t1 - t0;
                    calls_neighborhood++;
                }
            }

            t_closest += ep_closest;
            t_weight_update += ep_weight;
            t_neighborhood_update += ep_neighborhood;

            // --- recalculate neighbors ---
            double ep_calc_nb = 0;
            if (do_calc_neighbors)
            {
                t0 = now_sec();
                calculate_neighbors();
                t1 = now_sec();
                ep_calc_nb = t1 - t0;
                t_calc_neighbors += ep_calc_nb;
                calls_calc_neighbors++;
            }

            double ep1 = now_sec();
            double ep_total = ep1 - ep0;

            // Print a summary every epoch (condense to every 10 if noisy)
            if (epoch % 10 == 0 || epoch == epochs - 1)
            {
                cout << "[epoch " << epoch << "] "
                     << "total=" << ep_total * 1000.0 << " ms | "
                     << "shuffle=" << t_shuffle * 1000.0 / (epoch + 1) << " ms avg | " // running avg
                     << "closest=" << ep_closest * 1000.0 << " ms | "
                     << "weight_upd=" << ep_weight * 1000.0 << " ms | "
                     << "nb_upd=" << ep_neighborhood * 1000.0 << " ms | "
                     << "calc_nb=" << ep_calc_nb * 1000.0 << " ms\n";
            }
        }

        double total_train = now_sec() - epoch_start;

        // --- Final summary ---
        cout << "\n========= TIMING SUMMARY =========\n";
        cout << "Total training time:      " << total_train * 1000.0 << " ms\n\n";
        cout << "  shuffle:                " << t_shuffle * 1000.0
             << " ms  (" << 100.0 * t_shuffle / total_train << "%)\n";
        cout << "  closest_neuron:         " << t_closest * 1000.0
             << " ms  (" << 100.0 * t_closest / total_train << "%)  "
             << calls_closest << " calls\n";
        cout << "    └─ col_distance_vec:  " << t_cn_dist * 1000.0
             << " ms  (" << 100.0 * t_cn_dist / t_closest << "% of closest)\n";
        cout << "    └─ compare/update:    " << t_cn_compare * 1000.0
             << " ms  (" << 100.0 * t_cn_compare / t_closest << "% of closest)\n";
        cout << "    └─ overhead/other:    "
             << (t_closest - t_cn_dist - t_cn_compare) * 1000.0
             << " ms  (" << 100.0 * (t_closest - t_cn_dist - t_cn_compare) / t_closest << "% of closest)\n";
        cout << "    └─ avg per call:      "
             << t_closest * 1e6 / calls_closest << " us\n";
        cout << "    └─ avg dist per call: "
             << t_cn_dist * 1e9 / (calls_closest * num_neurons) << " ns/neuron\n";
        cout << "  weight update:          " << t_weight_update * 1000.0
             << " ms  (" << 100.0 * t_weight_update / total_train << "%)\n";
        cout << "  neighborhood update:    " << t_neighborhood_update * 1000.0
             << " ms  (" << 100.0 * t_neighborhood_update / total_train << "%)  "
             << calls_neighborhood << " calls\n";
        cout << "  calculate_neighbors:    " << t_calc_neighbors * 1000.0
             << " ms  (" << 100.0 * t_calc_neighbors / total_train << "%)  "
             << calls_calc_neighbors << " calls\n";
        cout << "  unaccounted:            "
             << (total_train - t_shuffle - t_closest - t_weight_update - t_neighborhood_update - t_calc_neighbors) * 1000.0 << " ms\n";
        cout << "===================================\n\n";

        // --- Cluster assignment ---
        t0 = now_sec();
        vector<int> clusters;
        for (const auto &x : data)
            clusters.push_back(closest_neuron(x));
        t1 = now_sec();
        cout << "[post] final cluster assignment: " << (t1 - t0) * 1000.0 << " ms\n";

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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: ./a.out <datafile>\n";
        return 1;
    }

    double t0, t1;

    t0 = now_sec();
    Matrix data = load_data(argv[1]);
    t1 = now_sec();
    cout << "[load] " << data.size() << " rows x " << data[0].size()
         << " cols in " << (t1 - t0) * 1000.0 << " ms\n\n";

    int num_neurons, epochs, update_neighbors_epoch, calculate_k_epoch, k_neighbors;
    double learning_rate, influence;
    int init_mode_input;

    auto read_int = [&](const string &prompt, int dv)
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
    auto read_double = [&](const string &prompt, double dv)
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

    num_neurons = read_int("Enter number of neurons", 10);
    epochs = read_int("Enter number of epochs", 100);
    learning_rate = read_double("Enter learning rate", 0.3);
    influence = read_double("Enter influence", 0.1);
    update_neighbors_epoch = read_int("Enter update_neighbors_epoch", 4);
    calculate_k_epoch = read_int("Enter calculate_k_epoch", 6);
    k_neighbors = read_int("Enter k_neighbors", 4);
    init_mode_input = read_int("Init mode (0=diagonal, 1=onepoint, 2=random)", 0);

    SOM::InitMode init_mode = (init_mode_input == 0)   ? SOM::DIAGONAL
                              : (init_mode_input == 1) ? SOM::ONEPOINT
                                                       : SOM::RANDOM;

    t0 = now_sec();
    SOM som(data, num_neurons, epochs, learning_rate, influence,
            update_neighbors_epoch, calculate_k_epoch, k_neighbors, init_mode);
    t1 = now_sec();
    cout << "\n[init] SOM construction (weights init): " << (t1 - t0) * 1000.0 << " ms\n\n";

    vector<int> clusters = som.train();

    cout << "\nCluster assignments:\n";
    map<int, int> count;
    for (int n : clusters)
        count[n]++;
    for (const auto &p : count)
        cout << p.first << ": " << p.second << "\n";

    return 0;
}