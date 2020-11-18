#include "TMath.h"

double MyErfc(double x, double x0, double width) {
  return TMath::Erfc((x - x0) / width);
};

