#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT 16
#define HiddenLayer1 32
#define HiddenLayer2 16
#define HiddenLayer3 8
#define OUTPUT 2
#define EPOCHS 30000

double learning_rate = 0.0007;
double mean_x[INPUT], std_x[INPUT];
double mean_y[OUTPUT], std_y[OUTPUT];

//nätverksstruktur
typedef struct {
    double Weight1[INPUT][HiddenLayer1]; double b1[HiddenLayer1];
    double Weight2[HiddenLayer1][HiddenLayer2]; double b2[HiddenLayer2];
    double Weight3[HiddenLayer2][HiddenLayer3]; double b3[HiddenLayer3];
    double Weight4[HiddenLayer3][OUTPUT]; double b4[OUTPUT];
} neuralnet;
//sätter random weight
double rand_weight(int fan_in) {
    return ((double)rand() / RAND_MAX - 0.5) * 2.0 / sqrt((double)fan_in);
}

//aktiveringsfunktion och dess dirivata
double act(double x) { return tanh(x); }
double dact(double x) { double t = tanh(x); return 1 - t * t; }

void normalize_X(double** X, int n_train, int total) {
    for (int j = 0; j < INPUT; j++) mean_x[j] = 0.0;
    for (int i = 0; i < n_train; i++)
        for (int j = 0; j < INPUT; j++)
            mean_x[j] += X[i][j];
    for (int j = 0; j < INPUT; j++) mean_x[j] /= n_train;

    for (int j = 0; j < INPUT; j++) std_x[j] = 0.0;
    for (int i = 0; i < n_train; i++)
        for (int j = 0; j < INPUT; j++) {
            double d = X[i][j] - mean_x[j];
            std_x[j] += d * d;
        }
    for (int j = 0; j < INPUT; j++) {
        std_x[j] = sqrt(std_x[j] / n_train);
        if (std_x[j] < 1e-12) std_x[j] = 1.0;
    }

    for (int i = 0; i < total; i++)
        for (int j = 0; j < INPUT; j++)
            X[i][j] = (X[i][j] - mean_x[j]) / std_x[j];
}

void normalize_Y(double** Y, int n_train) {
    for (int k = 0; k < OUTPUT; k++) {
        mean_y[k] = 0.0;
        for (int i = 0; i < n_train; i++) mean_y[k] += Y[i][k];
        mean_y[k] /= n_train;

        std_y[k] = 0.0;
        for (int i = 0; i < n_train; i++) {
            double d = Y[i][k] - mean_y[k];
            std_y[k] += d * d;
        }
        std_y[k] = sqrt(std_y[k] / n_train);
        if (std_y[k] < 1e-12) std_y[k] = 1.0;
    }
}

void Denormalize_y(double* y_norm, double* y_out) {
    for (int k = 0; k < OUTPUT; k++) y_out[k] = y_norm[k] * std_y[k] + mean_y[k];
}

void init_network(neuralnet* neuralnet) {
    srand((unsigned int)time(NULL));

    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HiddenLayer1; j++) neuralnet->Weight1[i][j] = rand_weight(INPUT);
    for (int j = 0; j < HiddenLayer1; j++) neuralnet->b1[j] = 0.0;

    for (int i = 0; i < HiddenLayer1; i++)
        for (int j = 0; j < HiddenLayer2; j++) neuralnet->Weight2[i][j] = rand_weight(HiddenLayer1);
    for (int j = 0; j < HiddenLayer2; j++) neuralnet->b2[j] = 0.0;

    for (int i = 0; i < HiddenLayer2; i++)
        for (int j = 0; j < HiddenLayer3; j++) neuralnet->Weight3[i][j] = rand_weight(HiddenLayer2);
    for (int j = 0; j < HiddenLayer3; j++) neuralnet->b3[j] = 0.0;

    for (int i = 0; i < HiddenLayer3; i++)
        for (int k = 0; k < OUTPUT; k++) neuralnet->Weight4[i][k] = rand_weight(HiddenLayer3);
    for (int k = 0; k < OUTPUT; k++) neuralnet->b4[k] = 0.0;
}

void forward(neuralnet* neuralnet, double* x, double* h1, double* h2, double* h3, double* out) {
    for (int j = 0; j < HiddenLayer1; j++) {
        double sum = neuralnet->b1[j];
        for (int i = 0; i < INPUT; i++) sum += x[i] * neuralnet->Weight1[i][j];
        h1[j] = act(sum);
    }
    for (int j = 0; j < HiddenLayer2; j++) {
        double sum = neuralnet->b2[j];
        for (int i = 0; i < HiddenLayer1; i++) sum += h1[i] * neuralnet->Weight2[i][j];
        h2[j] = act(sum);
    }
    for (int j = 0; j < HiddenLayer3; j++) {
        double sum = neuralnet->b3[j];
        for (int i = 0; i < HiddenLayer2; i++) sum += h2[i] * neuralnet->Weight3[i][j];
        h3[j] = act(sum);
    }
    for (int k = 0; k < OUTPUT; k++) {
        double sum = neuralnet->b4[k];
        for (int j = 0; j < HiddenLayer3; j++) sum += h3[j] * neuralnet->Weight4[j][k];
        out[k] = sum;
    }
}

//tränar datan framåt och bakåt
void train_step(neuralnet* neuralnet, double* x, double* y_true) {
    double h1[HiddenLayer1], h2[HiddenLayer2], h3[HiddenLayer3], out[OUTPUT];
    forward(neuralnet, x, h1, h2, h3, out);

    //Back propagation
    double y_norm[OUTPUT];
    for (int k = 0; k < OUTPUT; k++) y_norm[k] = (y_true[k] - mean_y[k]) / std_y[k];

    double delta_out[OUTPUT];
    for (int k = 0; k < OUTPUT; k++) delta_out[k] = out[k] - y_norm[k];

   
    double delta_h3[HiddenLayer3];
    for (int j = 0; j < HiddenLayer3; j++) {
        double grad = 0.0;
        for (int k = 0; k < OUTPUT; k++) grad += delta_out[k] * neuralnet->Weight4[j][k];
        delta_h3[j] = grad * dact(h3[j]);
    }

    double delta_h2[HiddenLayer2];
    for (int j = 0; j < HiddenLayer2; j++) {
        double grad = 0.0;
        for (int k = 0; k < HiddenLayer3; k++) grad += delta_h3[k] * neuralnet->Weight3[j][k];
        delta_h2[j] = grad * dact(h2[j]);
    }

    double delta_h1[HiddenLayer1];
    for (int j = 0; j < HiddenLayer1; j++) {
        double grad = 0.0;
        for (int k = 0; k < HiddenLayer2; k++) grad += delta_h2[k] * neuralnet->Weight2[j][k];
        delta_h1[j] = grad * dact(h1[j]);
    }

    //uppdatering av vikter och bias
    for (int j = 0; j < HiddenLayer3; j++)
        for (int k = 0; k < OUTPUT; k++) neuralnet->Weight4[j][k] -= learning_rate * h3[j] * delta_out[k];
    for (int k = 0; k < OUTPUT; k++) neuralnet->b4[k] -= learning_rate * delta_out[k];

    for (int i = 0; i < HiddenLayer2; i++)
        for (int j = 0; j < HiddenLayer3; j++) neuralnet->Weight3[i][j] -= learning_rate * h2[i] * delta_h3[j];
    for (int j = 0; j < HiddenLayer3; j++) neuralnet->b3[j] -= learning_rate * delta_h3[j];

    for (int i = 0; i < HiddenLayer1; i++)
        for (int j = 0; j < HiddenLayer2; j++) neuralnet->Weight2[i][j] -= learning_rate * h1[i] * delta_h2[j];
    for (int j = 0; j < HiddenLayer2; j++) neuralnet->b2[j] -= learning_rate * delta_h2[j];

    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HiddenLayer1; j++) neuralnet->Weight1[i][j] -= learning_rate * x[i] * delta_h1[j];
    for (int j = 0; j < HiddenLayer1; j++) neuralnet->b1[j] -= learning_rate * delta_h1[j];
}
//ser till att värdet förbättras
double validation_loss(neuralnet* neuralnet, double** X_val, double** Y_val, int n_val) {
    double mse = 0.0;
    double h1[HiddenLayer1], h2[HiddenLayer2], h3[HiddenLayer3], out[OUTPUT];
    for (int i = 0; i < n_val; i++) {
        forward(neuralnet, X_val[i], h1, h2, h3, out);
        for (int k = 0; k < OUTPUT; k++) {
            double y_norm = (Y_val[i][k] - mean_y[k]) / std_y[k];
            double diff = out[k] - y_norm;
            mse += diff * diff;
        }
    }
    return mse / (n_val * OUTPUT);
}

//utvärderar nätverket
void evaluate(neuralnet* neuralnet, double** X, double** Y, int n) {
    double mse[OUTPUT] = { 0.0 };
    double h1[HiddenLayer1], h2[HiddenLayer2], h3[HiddenLayer3], out[OUTPUT], y_out_denorm[OUTPUT];
    for (int i = 0; i < n; i++) {
        forward(neuralnet, X[i], h1, h2, h3, out);
        Denormalize_y(out, y_out_denorm);
        for (int k = 0; k < OUTPUT; k++) {
            double diff = y_out_denorm[k] - Y[i][k];
            mse[k] += diff * diff;
        }
    }
    for (int k = 0; k < OUTPUT; k++) mse[k] /= n;
    printf("Test MSE Compressor = %.12e (denormalized)\n", mse[0]);
    printf("Test MSE Turbine    = %.12e (denormalized)\n", mse[1]);
}

int main() {
    FILE* f = NULL;
    if (fopen_s(&f, "maintenance.txt", "r") != 0) {
        printf("Error: could not open maintenance.txt\n");
        return 1;
    }

    int total = 11934;
    double** X = malloc(total * sizeof(double*));
    double** Y = malloc(total * sizeof(double*));
    for (int i = 0; i < total; i++) {
        X[i] = malloc(INPUT * sizeof(double));
        Y[i] = malloc(OUTPUT * sizeof(double));
        for (int j = 0; j < INPUT; j++) fscanf_s(f, "%lf", &X[i][j]);
        for (int k = 0; k < OUTPUT; k++) fscanf_s(f, "%lf", &Y[i][k]);
    }
    fclose(f);

    srand(time(NULL));
    for (int i = total - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        double* tmpX = X[i]; X[i] = X[j]; X[j] = tmpX;
        double* tmpY = Y[i]; Y[i] = Y[j]; Y[j] = tmpY;
    }

    int n_train = total * 0.5;
    int n_val = total * 0.25;
    int n_test = total - n_train - n_val;

    double** X_train = X;
    double** Y_train = Y;
    double** X_val = X + n_train;
    double** Y_val = Y + n_train;
    double** X_test = X + n_train + n_val;
    double** Y_test = Y + n_train + n_val;

    normalize_X(X, n_train, total);
    normalize_Y(Y, n_train);

    neuralnet neuralnet;
    init_network(&neuralnet);

    double best_val_loss = 1e9;
    int patience_counter = 0;

    for (int e = 0; e < EPOCHS; e++) {
        for (int i = 0; i < n_train; i++) train_step(&neuralnet, X_train[i], Y_train[i]);
        if (e % 50 == 0) {
            double val_loss = validation_loss(&neuralnet, X_val, Y_val, n_val);
            printf("Epoch %d - Val Loss: %.8f - Learning Rate: %.8f\n", e, val_loss, learning_rate);
            if (val_loss < best_val_loss - 1e-8) {
                best_val_loss = val_loss;
                patience_counter = 0;
            }
            else {
                patience_counter++;
                if (patience_counter >= 10) {
                    learning_rate *= 0.95;
                    printf("Learning rate decreased to %.8f due to no improvement in val loss\n", learning_rate);
                    patience_counter = 0;
                }
            }
        }
    }

    evaluate(&neuralnet, X_test, Y_test, n_test);
    return 0;
}
