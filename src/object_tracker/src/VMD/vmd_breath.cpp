#include "VMD.h"
#include "data.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <Eigen/Eigen>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

int main(){
    int Fs = 1000;
    int sig_length = candidate_data.size();
    vectord sig_amp(sig_length);
    int i = 0;
    for (i=0;i<sig_length;i++)
    {
        sig_amp[i] = std::abs(candidate_data.at(i));
    }
    
	const double alpha = 50.0, tau = 0, tol = 1e-7, eps = 2.2204e-16;
	const int K = 4, DC = 0, init = 1;
	MatrixXd u, omega;
	MatrixXcd u_hat;
	
	VMD(u, u_hat, omega, sig_amp, alpha, tau, K, DC, init, tol, eps);

	Eigen::VectorXd new_sig = u.row(0);
	Eigen::VectorXd new_sig1 = new_sig.array() - new_sig.mean();

	//fft
	Eigen::FFT<double> fft;
    Eigen::VectorXcd Y(sig_length);
    fft.fwd(Y, new_sig1);

    int half_length = sig_length/2;
    int ignored_freq_bins =  int(0.13/1000 * sig_length) + 1;

    double P1[half_length - ignored_freq_bins];
    double max_val = -1;
    int max_idx = -1;
    for(int j=ignored_freq_bins;j<half_length;j++) 
    {
        P1[j-ignored_freq_bins] = std::abs(Y[j]); 
        if (P1[j-ignored_freq_bins] > max_val)
        {
            max_val = P1[j-ignored_freq_bins];
            max_idx = j;
        }
    }

    float bpm = 60 * float(max_idx) / float(sig_length) * Fs; 
    std::cout<< bpm;
    return 0;
}
