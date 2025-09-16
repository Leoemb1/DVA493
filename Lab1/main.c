#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT 16
#define HIDDEN 128
#define OUTPUT 2
#define EPOCHS 50000
#define LR 1e-5

// -------------------- GLOBAL ARRAYS --------------------
double W1[INPUT][HIDDEN];
double b1[HIDDEN];
double W2[HIDDEN][OUTPUT];
double b2[OUTPUT];

double hidden[HIDDEN];
double out[OUTPUT];

double mean_y[OUTPUT];
double std_y[OUTPUT];

// -------------------- ACTIVATION --------------------
double relu(double x) {
    return x > 0 ? x : 0;
}
double relu_deriv(double x) {
    return x > 0 ? 1 : 0;
}

// -------------------- FORWARD PASS --------------------
void forward(double* x) {
    for (int j = 0; j < HIDDEN; j++) {
        double sum = b1[j];
        for (int i = 0; i < INPUT; i++)
            sum += x[i] * W1[i][j];
        hidden[j] = relu(sum);
    }

    for (int k = 0; k < OUTPUT; k++) {
        double sum = b2[k];
        for (int j = 0; j < HIDDEN; j++)
            sum += hidden[j] * W2[j][k];
        out[k] = sum;
    }
}

// -------------------- TRAINING STEP --------------------
void train_step(double* x, double* y) {
    forward(x);

    double d_out[OUTPUT];
    for (int k = 0; k < OUTPUT; k++)
        d_out[k] = out[k] - y[k];

    double d_hidden[HIDDEN];
    for (int j = 0; j < HIDDEN; j++) {
        double grad = 0;
        for (int k = 0; k < OUTPUT; k++)
            grad += d_out[k] * W2[j][k];
        d_hidden[j] = grad * relu_deriv(hidden[j]);
    }

    // update W2, b2
    for (int j = 0; j < HIDDEN; j++)
        for (int k = 0; k < OUTPUT; k++)
            W2[j][k] -= LR * d_out[k] * hidden[j];
    for (int k = 0; k < OUTPUT; k++)
        b2[k] -= LR * d_out[k];

    // update W1, b1
    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HIDDEN; j++)
            W1[i][j] -= LR * d_hidden[j] * x[i];
    for (int j = 0; j < HIDDEN; j++)
        b1[j] -= LR * d_hidden[j];
}

// -------------------- LOSS --------------------
double mse_loss(double* pred, double* target) {
    double loss = 0;
    for (int k = 0; k < OUTPUT; k++) {
        double diff = pred[k] - target[k];
        loss += diff * diff;
    }
    return loss / OUTPUT;
}

// -------------------- DATA LOADING --------------------
int load_data(const char* fname, double*** X, double*** Y) {
    FILE* f = fopen(fname, "r");
    if (!f) { perror("File open failed"); exit(1); }

    int capacity = 1000, n = 0;
    *X = malloc(capacity * sizeof(double*));
    *Y = malloc(capacity * sizeof(double*));

    while (!feof(f)) {
        double* x = malloc(INPUT * sizeof(double));
        double* y = malloc(OUTPUT * sizeof(double));
        int read = 0;

        for (int i = 0; i < INPUT; i++) read += fscanf_s(f, "%lf", &x[i]);
        for (int j = 0; j < OUTPUT; j++) read += fscanf_s(f, "%lf", &y[j]);

        if (read == INPUT + OUTPUT) {
            if (n >= capacity) {
                capacity *= 2;
                *X = realloc(*X, capacity * sizeof(double*));
                *Y = realloc(*Y, capacity * sizeof(double*));
            }
            (*X)[n] = x;
            (*Y)[n] = y;
            n++;
        }
        else {
            free(x); free(y);
            break;
        }
    }
    fclose(f);
    return n;
}

// -------------------- NORMALISERA Y --------------------
void normalize_Y(double** Y, int n_train, int n_samples) {
    for (int k = 0; k < OUTPUT; k++) mean_y[k] = 0.0;
    for (int i = 0; i < n_train; i++)
        for (int k = 0; k < OUTPUT; k++)
            mean_y[k] += Y[i][k];
    for (int k = 0; k < OUTPUT; k++)
        mean_y[k] /= n_train;

    for (int k = 0; k < OUTPUT; k++) std_y[k] = 0.0;
    for (int i = 0; i < n_train; i++)
        for (int k = 0; k < OUTPUT; k++) {
            double d = Y[i][k] - mean_y[k];
            std_y[k] += d * d;
        }
    for (int k = 0; k < OUTPUT; k++)
        std_y[k] = sqrt(std_y[k] / n_train);

    for (int i = 0; i < n_samples; i++)
        for (int k = 0; k < OUTPUT; k++)
            Y[i][k] = (Y[i][k] - mean_y[k]) / std_y[k];
}

// -------------------- MAIN --------------------
int main() {
    srand((unsigned)time(NULL));

    double** X; double** Y;
    int n_samples = load_data("maintenance.txt", &X, &Y);
    printf("Loaded %d samples\n", n_samples);

    int n_train = (int)(0.8 * n_samples);
    int n_test = n_samples - n_train;

    // normalize Y using only train
    normalize_Y(Y, n_train, n_samples);

    // He-init
    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HIDDEN; j++)
            W1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / INPUT);
    for (int j = 0; j < HIDDEN; j++) b1[j] = 0.0;
    for (int j = 0; j < HIDDEN; j++)
        for (int k = 0; k < OUTPUT; k++)
            W2[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / HIDDEN);
    for (int k = 0; k < OUTPUT; k++) b2[k] = 0.0;

    // index for shuffle
    int* indices = malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = i;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // shuffle
        for (int i = n_train - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        double total_loss = 0;
        for (int idx = 0; idx < n_train; idx++) {
            int i = indices[idx];
            train_step(X[i], Y[i]);
            forward(X[i]);
            total_loss += mse_loss(out, Y[i]);
        }
        total_loss /= n_train;

        if (epoch % 100 == 0)
            printf("Epoch %d, Train MSE = %.6e\n", epoch, total_loss);
    }
    free(indices);

    // evaluate test
    double mse_comp = 0, mse_turb = 0;
    for (int i = n_train; i < n_samples; i++) {
        forward(X[i]);
        double y_pred0 = out[0] * std_y[0] + mean_y[0];
        double y_pred1 = out[1] * std_y[1] + mean_y[1];
        double y_true0 = Y[i][0] * std_y[0] + mean_y[0];
        double y_true1 = Y[i][1] * std_y[1] + mean_y[1];

        double err0 = y_pred0 - y_true0;
        double err1 = y_pred1 - y_true1;
        mse_comp += err0 * err0;
        mse_turb += err1 * err1;
    }
    mse_comp /= n_test;
    mse_turb /= n_test;

    printf("\n--- Final Results ---\n");
    printf("Test MSE compressor: %.2e\n", mse_comp);
    printf("Test MSE turbine:    %.2e\n", mse_turb);

    return 0;
}
