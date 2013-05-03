#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include<iostream>
#include<string>

//Sample code written for detecting and some rectifications of 1D barcode

using namespace cv;
using namespace std;

RNG rng(12345);

{

  Mat src, src_gray;
	Mat grad_0,grad_45,grad_90,grad_135;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/// Load an image
	src = imread( argv[1] );
	Mat src_rect = src;

	if( !src.data )
	{ return -1; }

	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Convert it to gray
	cvtColor( src, src_gray, COLOR_RGB2GRAY );

	/// Create window
	namedWindow( window_name, WINDOW_AUTOSIZE );

	/// Generate grad_x and grad_y
	//	addWeighted( abs_grad_x,-2.67 , abs_grad_y, 0.67, 0, grad_90 );
	//	addWeighted( abs_grad_x,0.67 , abs_grad_y, 2.67, 0, grad_45 );
	//	addWeighted( abs_grad_x,0.67 , abs_grad_y, -2.67, 0, grad_135 );
	//	addWeighted( abs_grad_x,0.5 , abs_grad_y, 0.5, 0, grad );
	//	imshow(window_name,grad);
	//		waitKey(0);

	Point2f src_center(src_gray.cols/2.0F, src_gray.rows/2.0F);
	double angle = 0;
	for(int i=0;i<1;i++)
	{
		cout << i << endl;
		Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
		Mat dst;
		warpAffine(src_gray, dst, rot_mat, grad.size());
		imshow(window_name,dst);
		//		waitKey(0);
		//		imshow(window_name,filtrd_gaus);
		//		waitKey(0);
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel( dst, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_x, abs_grad_x );

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel( dst, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_y, abs_grad_y );


		//temporary work
		Mat kernel1;
		Point anchor1;
		double delta2;
		int ddepth2;
		int kernel_size1;  

		anchor1 = Point( -1, -1 );
		delta2 = 0;
		ddepth2 = -1;
		
		//Sobel kernels
		float data1[3][3] = {{0,-1,-2},{1,0,-1},{2,1,0}}; //45 degree
		float data2[3][3] = {{0,1,2},{-1,0,1},{-2,-1,0}}; //-45 degree
		//		kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
		Mat temp_out1;
		Mat temp_out2;
		kernel1 = Mat(3, 3, CV_32F, &data1);  
		filter2D(dst,temp_out1, ddepth2 , kernel1, anchor1, delta2, BORDER_DEFAULT );
		kernel1 = Mat(3, 3, CV_32F, &data2);  
		filter2D(dst,temp_out2, ddepth2 , kernel1, anchor1, delta2, BORDER_DEFAULT );
		Mat temp_out = temp_out1 + temp_out2;
		imwrite("temp_dst.jpg",dst);
		imwrite("temp_out_x.jpg",temp_out);
		imwrite("temp_sobel_x.jpg",abs_grad_x);

		/// Total Gradient (approximate)
		Mat grad_1;
		addWeighted( abs_grad_x,1 , abs_grad_y, -1, 0, grad_1 );
		grad_1 = temp_out;

		equalizeHist(grad_1,grad);

		Mat kernel;
		Point anchor;
		double delta1;
		int ddepth1;
		kernel = Mat::ones( 15, 15, CV_32F )/ (float)(225);
		anchor = Point( -1, -1 );
		delta = 0;
		ddepth = -1;
		cout << "blah" << endl;

		ostringstream strs;
		strs << angle;
		string out_filtrd = "./output/cam_filtrd_" + strs.str() + ".jpg";
		string out_grad = "./output/cam_grad_" + strs.str() + ".jpg";
		string out_contour = "./output/cam_contour_" + strs.str() + ".jpg";

		cout << out_filtrd << endl;
		cout << out_grad << endl;
		angle += 45;
		Mat filtrd, filtrd_gaus;
		filter2D(grad, filtrd, ddepth1 , kernel, anchor, delta1, BORDER_DEFAULT );
		blur( filtrd, filtrd_gaus, Size( 5, 5), Point(-1,-1) );
		cout << "tooo" << endl;
		//		imshow(window_name,filtrd_gaus);
		//		waitKey(0);

		imwrite(out_filtrd,filtrd_gaus);
		imwrite(out_grad,grad);
		cout << "FU" << endl;

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		//		findContours( filtrd_gaus, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
		Mat drawing = Mat::zeros( filtrd.size(), CV_8UC3 );
		cout << contours.size() << endl;

		/// Find the convex hull object for each contour
		vector<vector<Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )
		{  convexHull( Mat(contours[i]), hull[i], false ); }


		for( int i = 0; i< contours.size(); i++ )
		{
			//	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			Scalar color = Scalar(  255,255,255 );
			drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
			//			drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		}
		//		namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
		cout << "done with rectangle" << endl;
		//		imshow(window_name,drawing);
		imwrite(out_contour,drawing);
		cout << "sdfdle" << endl;

		/*
		// fitting the barcode into a square box
		vector<Point> not_a_rect_shape;
		not_a_rect_shape.push_back(Point(712, 417));
		not_a_rect_shape.push_back(Point(1510,395));
		not_a_rect_shape.push_back(Point(1570,600));
		not_a_rect_shape.push_back(Point(712,625));

		const Point* point = &not_a_rect_shape[0];
		int n = (int)not_a_rect_shape.size();
		Mat draw = src_rect.clone();
		polylines(draw, &point, &n, 1, true, Scalar(0,0 ,255), 3, CV_AA);
		imwrite("draw.jpg", draw);
		RotatedRect box = minAreaRect(cv::Mat(not_a_rect_shape));
		Point2f pts[4];

		box.points(pts);

		cv::Point2f src_vertices[3];
		src_vertices[0] = pts[0];
		src_vertices[1] = pts[1];
		src_vertices[2] = pts[3];


		Point2f dst_vertices[3];
		dst_vertices[0] = Point(0, 0);
		dst_vertices[1] = Point(box.boundingRect().width-1, 0);
		dst_vertices[2] = Point(0, box.boundingRect().height-1);

		Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

		cv::Mat rotated;
		cv::Size size(box.boundingRect().width, box.boundingRect().height);
		warpAffine(src, rotated, warpAffineMatrix, size, INTER_LINEAR, BORDER_CONSTANT);

		 */
		Mat otsu_out;
		threshold(filtrd_gaus,otsu_out,128,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
		imwrite("./output/otsu_out.jpg",otsu_out);

	}
	//	imshow( window_name, filtrd );
	//	waitKey(0);

	return 0;
}
