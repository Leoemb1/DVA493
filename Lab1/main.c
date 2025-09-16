#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================
// Hyperparameters
// =========================
#define INPUTS 16
#define H1 512
#define H2 256
#define H3 128
#define H4 64
#define H5 32
#define OUTPUTS 2
#define EPOCHS 1000
#define LR 0.0001
#define MAX_SAMPLES 10000  // adjust if your dataset is bigger

// =========================
// Activation functions
// =========================
double tanh_act(double x) { return tanh(x); }
double tanh_deriv(double x) { double t = tanh(x); return 1 - t*t; }

// =========================
// Random init
// =========================
double rand_weight() {
    return ((double)rand() / RAND_MAX) * 0.2 - 0.1;
}

// =========================
// Buffers for forward pass
// =========================
double z1[H1], a1[H1];
double z2[H2], a2[H2];
double z3[H3], a3[H3];
double z4[H4], a4[H4];
double z5[H5], a5[H5];
double z6[OUTPUTS], out[OUTPUTS];

// Weights and biases
double W1[H1][INPUTS], b1[H1];
double W2[H2][H1], b2[H2];
double W3[H3][H2], b3[H3];
double W4[H4][H3], b4[H4];
double W5[H5][H4], b5[H5];
double W6[OUTPUTS][H5], b6[OUTPUTS];

// =========================
// Network init
// =========================
void init_network() {
    srand(time(NULL));
    for (int i=0;i<H1;i++){for(int j=0;j<INPUTS;j++) W1[i][j]=rand_weight(); b1[i]=0;}
    for (int i=0;i<H2;i++){for(int j=0;j<H1;j++) W2[i][j]=rand_weight(); b2[i]=0;}
    for (int i=0;i<H3;i++){for(int j=0;j<H2;j++) W3[i][j]=rand_weight(); b3[i]=0;}
    for (int i=0;i<H4;i++){for(int j=0;j<H3;j++) W4[i][j]=rand_weight(); b4[i]=0;}
    for (int i=0;i<H5;i++){for(int j=0;j<H4;j++) W5[i][j]=rand_weight(); b5[i]=0;}
    for (int i=0;i<OUTPUTS;i++){for(int j=0;j<H5;j++) W6[i][j]=rand_weight(); b6[i]=0;}
}

// =========================
// Forward pass
// =========================
void forward(double x[INPUTS]) {
    for(int i=0;i<H1;i++){z1[i]=b1[i]; for(int j=0;j<INPUTS;j++) z1[i]+=W1[i][j]*x[j]; a1[i]=tanh_act(z1[i]);}
    for(int i=0;i<H2;i++){z2[i]=b2[i]; for(int j=0;j<H1;j++) z2[i]+=W2[i][j]*a1[j]; a2[i]=tanh_act(z2[i]);}
    for(int i=0;i<H3;i++){z3[i]=b3[i]; for(int j=0;j<H2;j++) z3[i]+=W3[i][j]*a2[j]; a3[i]=tanh_act(z3[i]);}
    for(int i=0;i<H4;i++){z4[i]=b4[i]; for(int j=0;j<H3;j++) z4[i]+=W4[i][j]*a3[j]; a4[i]=tanh_act(z4[i]);}
    for(int i=0;i<H5;i++){z5[i]=b5[i]; for(int j=0;j<H4;j++) z5[i]+=W5[i][j]*a4[j]; a5[i]=tanh_act(z5[i]);}
    for(int i=0;i<OUTPUTS;i++){z6[i]=b6[i]; for(int j=0;j<H5;j++) z6[i]+=W6[i][j]*a5[j]; out[i]=z6[i];}
}

// =========================
// Backpropagation
// =========================
void backward(double x[INPUTS], double y[OUTPUTS]) {
    double delta6[OUTPUTS];
    for(int i=0;i<OUTPUTS;i++) delta6[i]=2*(out[i]-y[i]);

    double delta5[H5];
    for(int i=0;i<H5;i++){double sum=0; for(int j=0;j<OUTPUTS;j++) sum+=W6[j][i]*delta6[j]; delta5[i]=sum*tanh_deriv(z5[i]);}
    double delta4[H4];
    for(int i=0;i<H4;i++){double sum=0; for(int j=0;j<H5;j++) sum+=W5[j][i]*delta5[j]; delta4[i]=sum*tanh_deriv(z4[i]);}
    double delta3[H3];
    for(int i=0;i<H3;i++){double sum=0; for(int j=0;j<H4;j++) sum+=W4[j][i]*delta4[j]; delta3[i]=sum*tanh_deriv(z3[i]);}
    double delta2[H2];
    for(int i=0;i<H2;i++){double sum=0; for(int j=0;j<H3;j++) sum+=W3[j][i]*delta3[j]; delta2[i]=sum*tanh_deriv(z2[i]);}
    double delta1[H1];
    for(int i=0;i<H1;i++){double sum=0; for(int j=0;j<H2;j++) sum+=W2[j][i]*delta2[j]; delta1[i]=sum*tanh_deriv(z1[i]);}

    for(int i=0;i<OUTPUTS;i++){for(int j=0;j<H5;j++) W6[i][j]-=LR*delta6[i]*a5[j]; b6[i]-=LR*delta6[i];}
    for(int i=0;i<H5;i++){for(int j=0;j<H4;j++) W5[i][j]-=LR*delta5[i]*a4[j]; b5[i]-=LR*delta5[i];}
    for(int i=0;i<H4;i++){for(int j=0;j<H3;j++) W4[i][j]-=LR*delta4[i]*a3[j]; b4[i]-=LR*delta4[i];}
    for(int i=0;i<H3;i++){for(int j=0;j<H2;j++) W3[i][j]-=LR*delta3[i]*a2[j]; b3[i]-=LR*delta3[i];}
    for(int i=0;i<H2;i++){for(int j=0;j<H1;j++) W2[i][j]-=LR*delta2[i]*a1[j]; b2[i]-=LR*delta2[i];}
    for(int i=0;i<H1;i++){for(int j=0;j<INPUTS;j++) W1[i][j]-=LR*delta1[i]*x[j]; b1[i]-=LR*delta1[i];}
}

// =========================
// Training
// =========================
void train(double X[][INPUTS], double Y[][OUTPUTS], int n_samples) {
    for(int epoch=0; epoch<EPOCHS; epoch++){
        double mse=0;
        for(int i=0;i<n_samples;i++){
            forward(X[i]);
            backward(X[i], Y[i]);
            for(int k=0;k<OUTPUTS;k++) mse += (out[k]-Y[i][k])*(out[k]-Y[i][k]);
        }
        mse /= (n_samples*OUTPUTS);
        if(epoch % 100 == 0) printf("Epoch %d, MSE = %.10f\n", epoch, mse);
    }
}

// =========================
// File loading
// =========================
int load_dataset(const char *filename, double X[][INPUTS], double Y[][OUTPUTS]) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { printf("Error: cannot open %s\n", filename); exit(1); }

    int n=0;
    while (!feof(fp) && n < MAX_SAMPLES) {
        double row[INPUTS+OUTPUTS];
        int count = 0;
        for(int j=0;j<INPUTS+OUTPUTS;j++) {
            if (fscanf(fp, "%lf", &row[j]) != 1) break;
            count++;
        }
        if (count == INPUTS+OUTPUTS) {
            for(int j=0;j<INPUTS;j++) X[n][j] = row[j];
            for(int j=0;j<OUTPUTS;j++) Y[n][j] = row[INPUTS+j];
            n++;
        }
    }
    fclose(fp);
    printf("Loaded %d samples from %s\n", n, filename);
    return n;
}

// =========================
// Main
// =========================
int main() {
    double X[MAX_SAMPLES][INPUTS];
    double Y[MAX_SAMPLES][OUTPUTS];

    int n_samples = load_dataset("maintenance.txt", X, Y);

    init_network();
    train(X, Y, n_samples);

    // Test on first sample
    forward(X[0]);
    printf("Prediction vs target:\n");
    printf("Pred: %.6f %.6f\n", out[0], out[1]);
    printf("True: %.6f %.6f\n", Y[0][0], Y[0][1]);
    return 0;
}
