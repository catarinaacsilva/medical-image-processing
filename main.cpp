#include <iostream>
#include "lib_od.h"

using namespace cv;
using namespace std;


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
		cout << "Image file could not be open !!" << std::endl;
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
    waitKey( 0 );

    // Smooth image
    Mat smooth_image;
    medianBlur(originalImage, smooth_image, 9);
    namedWindow("Averaging Filter 9 x 9 - 1 Iter", WINDOW_AUTOSIZE);
    imshow( "Averaging Filter 9 x 9 - 1 Iter", smooth_image );
	waitKey( 0 );

    // Binary image
    Mat binary_image;
    
    // Set threshold and maxValue
    double thresh = 0;
    double maxValue = 255;
    threshold(smooth_image, binary_image, thresh, maxValue, THRESH_OTSU);
    namedWindow("THRESH OTSU", WINDOW_AUTOSIZE );
    imshow("THRESH OTSU", binary_image);
    waitKey( 0 );


    //After binarization is necessary reduce noise, again
    Mat smooth_image_2;
    medianBlur(binary_image, smooth_image_2, 9);
    namedWindow("Averaging Filter 9 x 9 - 1 Iter", WINDOW_AUTOSIZE);
    imshow( "Averaging Filter 9 x 9 - 1 Iter", smooth_image );
	waitKey( 0 );

    //canny
    int lowThreshold = 0;
    const int radio = 3;
    const int kernel_size = 3;
    Canny(smooth_image_2, smooth_image_2, lowThreshold, lowThreshold*radio , kernel_size);
    namedWindow("Canny", WINDOW_AUTOSIZE);
    imshow("Canny", smooth_image_2);
    waitKey(0);


    destroyAllWindows();
    return 0;
}