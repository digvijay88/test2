#include<iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include<iostream>
#include<string>
#include<vector>

using namespace std;
using namespace cv;

typedef unsigned int TUInt;
typedef int TInt;
typedef uint8_t TUInt8;
typedef int8_t TInt8;
typedef uint16_t TUInt16;
typedef int16_t TInt16;
typedef uint32_t TUInt32;
typedef int32_t TInt32;
typedef uint64_t TUInt64;
typedef int64_t TInt64;

//simple cv structures
typedef cv::Point_<int> TPointInt;
typedef cv::Point_<TUInt> TPointUInt;
typedef cv::Point_<double> TPointDouble;
typedef cv::Size_<int> TSizeInt;
typedef cv::Size_<TUInt> TSizeUInt;
typedef cv::Size_<double> TSizeDouble;
typedef cv::Rect_<int> TRectInt;
typedef cv::Rect_<TUInt> TRectUInt;
typedef cv::Rect_<double> TRectDouble;
//matrices
typedef cv::Mat_<int> TMatrixInt;
typedef cv::Mat_<TUInt> TMatrixUInt;
typedef cv::Mat_<TUInt8> TMatrixUInt8;
typedef cv::Mat_<float> TMatrixFloat;
typedef cv::Mat_<double> TMatrixDouble;
typedef cv::Mat_<bool> TMatrixBool;


const double PI=(4*atan(1.0));
int nOrientations = 18;
RNG rng(12345);

class Cell
{
  public:
		/** boundaries of the cell */
		TRectInt box;
		/** Histogram of orientations for this cell */
		vector<TUInt> orientationHistogram;
		/** Histogram of orientations for this cell */
		vector<TUInt> weightedOrientationHistogram;
	private:
		/** Dominant orientation of this cell, value is set by findDominantOrientation(), which must be run before reading. */
		int dominantOrientation_;
		/** Entropy of this cell, value is set by calculateEntropy(), which must be run before reading. */
		double entropy_;
		/** Number of pixels voting in the orientation histogram */
		TUInt nVoters_;
		/** Maximum entropy allowed */
		double maxEntropy_;
		/** Number of possible orientations */
		TUInt nOrientations_;
	public:
		/**^M
		 * Constructor^M
		 * @param[in] aBox cell area.^M
		 * @param[in] nOrientations number of orientation bins.^M
		 * @param[in] maximum entropy allowed^M
		 */
		Cell(const TRectInt &aBox, TUInt nOrientations, double maxEntropy);
		/**^M
		 * Clears the voters the orientation histogram, and the isSet flag.^M
		 */
		void reset();
		/**^M
		 * Adds a new pixel vote^M
		 * @param[in] orientation orientation of the pixel^M
		 * @param[in] magnitude of this pixel
		 */
		void addVoter(TUInt8 orientation, TUInt8 magnitude);
		/**^M
		 * Returns the dominant orientation in this cell and sets the dominantOrientation value.^M
		 * @return dominant orientation^M
		 */
		int dominantOrientation();
		/**^M
		 * Calculates the entropy of this cell, and sets the entropy value.^M
		 * @return entropy
		 */
		double entropy();
		/**^M
		 * Returns true is this cell has low entropy
		 */
		inline bool hasLowEntropy() {return ( entropy() < maxEntropy_ ); };
		/**^M
		 * Returns true is this cell has sufficient number of voters^M
		 */
		inline bool hasEnoughVoters() const {return ( nVoters_ > ((TUInt) box.area() >> 2) ); };
		/**^M
		 * True if this cell passes the tests for further consideration^M
		 * @return true if the cell has both low entropy and sufficient number of voters^M
		 */
		inline bool shouldBeConsidered() {return hasLowEntropy() && hasEnoughVoters(); };
		/**^M
		 * Returns the number of voting pixels
		 */
		inline TUInt nVoters() {return nVoters_; };
		/**^M
		 * Returns the center of the cell
		 */
		inline TPointInt center() {return (box.tl() + box.br()) * .5; };
};

int main(int, char** argv)
{

	Mat src,gray,frame_gray;
	src = imread(argv[1]); 	//Loaded the input image;
	//size_ = src.size();
	// Initialised the image containers (not required for now)

	//Starts the loop
	Mat frame = src; 	//current frame
	//Processing the frame
	////Prepare the barcode engine

	////now processing starts
	////// convert to gray
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//// Locate the list of barcodes
	vector<int /*Vote*/> orientationModes;	//correct
	//Get votes from pixels with gradients above threshold
	//getOrientationCandidates(orientationModes);
	//Calculate gradients
	// ->image_.update();
	//// some subsampling
	//// calculate scharr gradients
	//TUInt M = frame_gray.size().height, N = frame_gray.size().width;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad_x,grad_y;
	//x gradient
	Scharr( frame_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//	convertScaleAbs( grad_x, grad_x );
	//y gradient
	Scharr( frame_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//	convertScaleAbs( grad_y, grad_y );

	imwrite("grad_x.jpg",grad_x);			
	imwrite("grad_y.jpg",grad_y);			
	cout << "got here" << endl;

	//calculatePolarGradients
	////create magnitude and orientation maps
	TUInt8 thresh = 20;
	static const int MIN_GRAD = -255;
	static const int MAX_GRAD = 255;
	TMatrixUInt8 gradientMagnitudeMap(MAX_GRAD - MIN_GRAD + 1, MAX_GRAD - MIN_GRAD + 1)  ;
	TMatrixUInt8 gradientOrientationMap(MAX_GRAD - MIN_GRAD + 1, MAX_GRAD - MIN_GRAD + 1) ;
	for(int i = 0; i < gradientMagnitudeMap.rows; i++)
		for(int j = 0; j < gradientMagnitudeMap.cols; j++)
			gradientMagnitudeMap(i,j) = 0;
	for(int i = 0; i < gradientOrientationMap.rows; i++)
		for(int j = 0; j < gradientOrientationMap.cols; j++)
			gradientOrientationMap(i,j) = 0;
	cout << gradientMagnitudeMap(0,0) << "sdsd" <<endl;
	int diNorm, djNorm;
	double angle, dTheta = 2 * PI / nOrientations;
	TUInt mag, thresh2 = (TUInt) thresh * (TUInt) thresh;
	for (int di = MIN_GRAD; di <= MAX_GRAD; di++)
	{   
		diNorm = di - MIN_GRAD;
		for (int dj = MIN_GRAD; dj <= MAX_GRAD; dj++)
		{   
			djNorm = dj - MIN_GRAD;
			mag = (TUInt) (di*di + dj*dj);
			//			cout << mag << " " << thresh2 << " "  << gradientMagnitudeMap.size() << endl;
			//			cout <<  (mag > thresh2 ? sqrt((double) (mag>>1)) : 0) << endl;
			//			cout << "got here" << endl;
			gradientMagnitudeMap(diNorm, djNorm) = (int) (mag > thresh2 ? sqrt((double) (mag>>1)) : 0);  //scaled to ensure it will fit in TUInt8.
			if (gradientMagnitudeMap(diNorm, djNorm))
			{   
				angle = atan2((double) di, (double) dj);
				gradientOrientationMap(diNorm, djNorm) = ((TUInt8) (angle / dTheta + 0.5 + nOrientations)) % nOrientations;
			}   
			else
				gradientOrientationMap(diNorm, djNorm) = nOrientations;
		}   
	}   
	///finally calculating polar gradients
	const TMatrixInt &iGrad = grad_x;
	const TMatrixInt &jGrad = grad_y;
	const TUInt M = iGrad.size().height, N = iGrad.size().width;
	TMatrixUInt8 absGrad;
	TMatrixUInt8 angGrad;
	const TMatrixUInt8 &magnitudeLookup = gradientMagnitudeMap;
	const TMatrixUInt8 &orientationLookup = gradientOrientationMap;
	const TInt *iGradData, *jGradData;
	TUInt8 *absGradData, *angGradData;
	TInt curDI, curDJ;
	const TInt *iGradRowBegin = iGrad[0], *iGradRowEnd = iGradRowBegin + N-1;
	const TInt *jGradRowBegin = jGrad[0];
	TUInt8 *absGradRowBegin = absGrad[0], *angGradRowBegin = angGrad[0];
	cout << gradientMagnitudeMap.size() << endl;
	TMatrixUInt8  M1(20,20);
	for(int i = 0; i < M1.rows; i++)
		for(int j = 0; j < M1.cols; j++)
			M1(i,j) = 1;
	//	cout << (int) M1(0,0) << "sd" << endl;

	for (TUInt i = 0; i < M-1;
			i++, iGradRowBegin += N, iGradRowEnd += N, jGradRowBegin += N, absGradRowBegin += N, angGradRowBegin += N)
	{   
		for (iGradData = iGradRowBegin, jGradData = jGradRowBegin, absGradData = absGradRowBegin, angGradData = angGradRowBegin;
				iGradData < iGradRowEnd; iGradData++, jGradData++, absGradData++, angGradData++)
		{   
			//look up the magnitude and orientation of gradient for these di/dj values
			curDI = *iGradData + 255;
			curDJ = *jGradData + 255;
			cout << curDI << " " << curDJ << " " << (int) gradientMagnitudeMap(curDI, curDJ) << "sdf" << endl;
			cout << (TUInt8) magnitudeLookup(curDI,curDJ) << endl;
			cout << "df"<< endl;
			*absGradData = (TUInt8) magnitudeLookup(curDI, curDJ);
			*angGradData = (TUInt8) orientationLookup(curDI, curDJ); //TODO: see if it is faster to look these up or calculate
		}   
	}  
	return 0;

	//Calculate the cell histograms
	vector<vector<Cell> > cells_;


}
