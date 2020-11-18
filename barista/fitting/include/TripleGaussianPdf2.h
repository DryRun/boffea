/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
  * This code was autogenerated by RooClassFactory                            * 
 *****************************************************************************/

#ifndef TRIPLEGAUSSIANPDF2
#define TRIPLEGAUSSIANPDF2

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
 
class TripleGaussianPdf2 : public RooAbsPdf {
  public:
  TripleGaussianPdf2() {} ; 
  TripleGaussianPdf2(const char *name, const char *title,
        RooAbsReal& _x,
        RooAbsReal& _mean,
        RooAbsReal& _s1,
        RooAbsReal& _s2,
        RooAbsReal& _s3,
        RooAbsReal& _ccore,
        RooAbsReal& _ctail);
  TripleGaussianPdf2(const TripleGaussianPdf2& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new TripleGaussianPdf2(*this,newname); }
  inline virtual ~TripleGaussianPdf2() { }

  //Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  //Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

  protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy s1 ;
  RooRealProxy s2 ;
  RooRealProxy s3 ;
  RooRealProxy ccore ;
  RooRealProxy ctail ;

  Double_t evaluate() const ;

  private:

  ClassDef(TripleGaussianPdf2,1) // Your description goes here...
};
 
#endif
