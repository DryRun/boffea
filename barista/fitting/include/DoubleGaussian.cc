#include "TMath.h"

double DoubleGaussian(double x, double mean, double s1, double s2, double c1) {
	return c1 * TMath::Gaus(x, mean, s1, kTRUE) + (1. - c1) * TMath::Gaus(x, mean, s2, kTRUE);
};