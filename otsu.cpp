#include <iostream>


#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace cv;


int main(int argc, char** argv){
    if( argc != 2 ){
		std::cout << "The name of the image file is missing !!" << std::endl;

        return -1;
    }

	Mat originalImage;

	originalImage = imread( argv[1], IMREAD_UNCHANGED );

	if( originalImage.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty

		std::cout << "Image file could not be open !!" << std::endl;

	    return -1;
	}

	if( originalImage.channels() > 1 )
	{
	    // Convert to a single-channel, intensity image

		cvtColor( originalImage, originalImage, COLOR_BGR2GRAY, 1 );
	}

    // Create window

	namedWindow( "Original Image", WINDOW_AUTOSIZE );

    // Display image

	imshow( "Original Image", originalImage );
    // Read image
    //Mat src = imread("img00.png", IMREAD_GRAYSCALE);
    Mat dst;
    
    // Set threshold and maxValue
    double thresh = 0;
    double maxValue = 255; 
    
    // Binary Threshold
    /*
    threshold(originalImage,dst, thresh, maxValue, THRESH_BINARY);
    namedWindow( "THRESH BINARY", WINDOW_AUTOSIZE );
    imshow("THRESH BINARY", dst);

    threshold(originalImage,dst, thresh, maxValue, THRESH_BINARY_INV);
    namedWindow( "THRESH BINARY INV", WINDOW_AUTOSIZE );
    imshow("THRESH BINARY INV", dst);

    threshold(originalImage,dst, thresh, maxValue, THRESH_TRUNC);
    namedWindow( "THRESH TRUNC", WINDOW_AUTOSIZE );
    imshow("THRESH TRUNC", dst);

    threshold(originalImage,dst, thresh, maxValue, THRESH_TOZERO);
    namedWindow( "THRESH TOZERO", WINDOW_AUTOSIZE );
    imshow("THRESH TOZERO", dst);

    threshold(originalImage,dst, thresh, maxValue, THRESH_TOZERO_INV);
    namedWindow( "THRESH TOZERO INV", WINDOW_AUTOSIZE );
    imshow("THRESH TOZERO INV", dst);
    */

    // otsu
    threshold(originalImage,dst, thresh, maxValue, THRESH_OTSU);
    namedWindow( "THRESH OTSU", WINDOW_AUTOSIZE );
    imshow("THRESH OTSU", dst);

    /*
    //binary + otsu
    threshold(originalImage,dst, thresh, maxValue, THRESH_BINARY+THRESH_OTSU);
    namedWindow( "THRESH BINARY + OTSU", WINDOW_AUTOSIZE );
    imshow("THRESH BINARY + OTSU", dst);
    */

    waitKey( 0 );
    destroyAllWindows();

    return 0;

}