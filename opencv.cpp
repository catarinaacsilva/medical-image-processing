#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


//eliminar o ruido
cv::Mat smoothing(cv::Mat originalImage){
    cv::Mat averagedImage_9;
	cv::medianBlur( originalImage, averagedImage_9, 9);
	cv::namedWindow("Averaging Filter 9 x 9 - 1 Iter", cv::WINDOW_AUTOSIZE);
    return averagedImage_9;
}



int main(int argc, char** argv){
    if( argc != 2 ){
		std::cout << "The name of the image file is missing !!" << std::endl;
        return -1;
    }

    cv::Mat originalImage;
	originalImage = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

	if( originalImage.empty() ){
		// NOT SUCCESSFUL : the data attribute is empty
		std::cout << "Image file could not be open !!" << std::endl;
	    return -1;
	}

	if( originalImage.channels() > 1 ){
	    // Convert to a single-channel, intensity image
		cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2GRAY, 1);
	}

    // Create window
	cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);

    // Display image
	cv::imshow("Original Image", originalImage);

    //Aplicar o filtro de media --> 9x9
    cv::Mat averagedImage_9;
    averagedImage_9 = smoothing(originalImage);
    cv::imshow( "Averaging Filter 9 x 9 - 1 Iter", averagedImage_9 );


	cv::waitKey( 0 );
	// Destroy the windows
	cv::destroyAllWindows();

	return 0;
}