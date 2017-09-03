//SRI RAMA JEYAM
//V Vinokanth
//04/04/17
//LaundroBot Project
//Imaging system class

#include <camera.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

#ifndef IMAGINGSYSTEM_H
#define IMAGINGSYSTEM_H

using namespace cv;
using namespace std;

class ImagingSystem
{
public:
    ImagingSystem();
    Camera topCamera; //Camera placed on the ceiling frame - top view
    Camera frontCamera; //Camera placed on the wall frame - front view
    Point findSegment(); //Returns the coordinate of the segment of cloth to be picked by the manipilator system
    void trackPoint(Point featurePoint); //Tracks a known point
    Mat maskSection(Mat src); //masks out a section of image to assist ironing

};

#endif // IMAGINGSYSTEM_H
