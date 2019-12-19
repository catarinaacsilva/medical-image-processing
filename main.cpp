#include <iostream>
#include "lib_od.h"

using namespace cv;
using namespace std;



//create structuring element
Mat structuring_element(){
    int morph_size = 3;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 4*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size));
    return element;
}


//reduce noise with morphologic operartions
Mat morph_opening(Mat imageInput){
    Mat image_dest;
    morphologyEx( imageInput, image_dest, MORPH_OPEN, structuring_element() ); // 1 iteração
    return image_dest;
}

Mat morph_grad(Mat imageInput){
    Mat image_dest;
    morphologyEx( imageInput, image_dest, MORPH_GRADIENT, structuring_element() );
    return image_dest;
}


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


    // Smooth image eliminar ruido
    Mat smooth_image;
    medianBlur(originalImage, smooth_image, 9);
    namedWindow("Averaging Filter 9 x 9 - 1 Iter", WINDOW_AUTOSIZE);
    imshow( "Averaging Filter 9 x 9 - 1 Iter", smooth_image );
	waitKey( 0 );


    // Smooth image --> morphological operations - opening
    Mat smooth_image_op;
    namedWindow("opening", WINDOW_AUTOSIZE);
    imshow( "opening", morph_opening(originalImage));
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
    imshow( "Averaging Filter 9 x 9 - 1 Iter", smooth_image_2 );
	waitKey( 0 );


    //canny
    Mat image_result;
    int lowThreshold = 0;
    const int radio = 3;
    const int kernel_size = 3;
    Canny(smooth_image_2, image_result, lowThreshold, lowThreshold*radio , kernel_size);
    namedWindow("Canny00", WINDOW_AUTOSIZE);
    imshow("Canny00", image_result);
    waitKey(0);

    Mat cannyImage;
    Canny( smooth_image_2, cannyImage, 100, 100*2 );
    namedWindow("canny01", WINDOW_AUTOSIZE);
    imshow( "canny01", cannyImage );
	waitKey( 0 );

    //grad morph
    Mat element = structuring_element();
    namedWindow("grad", WINDOW_AUTOSIZE);
    imshow("grad", morph_grad(smooth_image_2));
    waitKey(0);


    //find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image_result,contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));


    //bounding box
    RNG rng(12345);
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ ){
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }
    Mat drawing = Mat::zeros( image_result.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours_poly, (int)i, color );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
        circle( drawing, centers[i], (int)radius[i], color, 2 );
    }
    namedWindow("grad", WINDOW_AUTOSIZE);
    imshow("grad", drawing);
    waitKey(0);
    

    destroyAllWindows();
    return 0;
}