//SRI RAMA JEYAM
//V Vinokanth
//31/03/17
//LaundroBot Project
//Ironing system class

/*This is the header file for namespace EdgesPolygons
 * This contains utility member functions for approximating polygons and detecting edges of images
 */

#include <opencv2/opencv.hpp>
#include <stdlib.h>

#ifndef EDGESPOLYGONS_H
#define EDGESPOLYGONS_H

using namespace cv;
using namespace std;

namespace EdgesPolygons
{
    //Member functions
    Mat detectEdges(Mat src); //Detects edges of the image using Sobel filter
    vector<Point>  approxOuterPolygon(Mat edge); //Returns an approximated vector of polygon coordintates
}

#endif // EDGESPOLYGONS_H
