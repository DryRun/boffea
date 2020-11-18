#include "TMath.h"

double TripleGaussian(double x, double mean, double s1, double s2, double s3, double aa, double bb) {
	// Enforce s1 < s2 < s3... not sure if it works
	if (s1 > s2 || s2 > s3) {
		return 1.e20;
	}
	double c1 = aa * TMath::Max(aa, bb) / (aa + bb);
	double c2 = bb * TMath::Max(aa, bb) / (aa + bb);
	double c3 = 1. - c1 - c2;
	return c1 * TMath::Gaus(x, mean, s1, kTRUE) + c2 * TMath::Gaus(x, mean, s2, kTRUE) + c3 * TMath::Gaus(x, mean, s3, kTRUE);
};
