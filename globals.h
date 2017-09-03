//SRI RAMA JEYAM
//V Vinokanth
//3/04/17
//LaundroBot Project
//Globals namespace

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#ifndef GLOBALS_H
#define GLOBALS_H

using namespace cv;

//This is namespace for globally used variables across the entire firmware
namespace GLOBALS
{
    namespace IRONING
    {
        extern Point homeIroningSystem;
        extern int sprayShoots; //Number of shoots of spray system
        extern int minLiquidLevel;
        extern int padSprayOffset; //This is the CAD offset dimesion of the center of iron pad and the spray nozzle (pixels)

        //IronPadLever position enum
        enum LEVER_POSITION
        {
            LEVER_UP,
            LEVER_DOWN
        };
    }


}

#endif // GLOBALS_H

