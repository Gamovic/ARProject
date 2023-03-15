using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ARExercise
{
    public class MarkerAR : FrameLoop
    {
        VideoCapture vCap;

        Matrix<float> intrinsic;
        Matrix<float> distortionCoeff;

        Matrix<float> rotationVector2 = new Matrix<float>(3, 1);
        Matrix<float> translationVector2 = new Matrix<float>(3, 1);
        Matrix<float> newRotMatrix2 = new Matrix<float>(3, 3);

        public MarkerAR()
        {
            vCap = new VideoCapture(1);
            //Mat image2 = CvInvoke.Imread("four.jpg");
            //CvInvoke.Imshow("hellowad", image2);

            intrinsic = new Matrix<float>(3, 3);
            distortionCoeff = new Matrix<float>(1, 5);
            //Read intrinsic and distortionCoeff from CameraCalibration (.json file)
            UtilityAR.ReadIntrinsicsFromFile(out intrinsic, out distortionCoeff);

            // Keep on capturing images
            //UtilityAR.CaptureLoop(new Size(7, 4), 1);
            //UtilityAR.CalibrateCamera(new Size(7, 4), true);

            // load image
            //Mat image = CvInvoke.Imread("capture_27.jpg");
            //Mat image2 = CvInvoke.Imread("capture_28.jpg");
            Mat image3 = CvInvoke.Imread("capture_1.jpg");
            // show image
            //CvInvoke.Imshow("hello", image);

            CvInvoke.Imshow("f", image3);

            // new gray image mat
            Mat grayImage = new Mat();

            // from color to gray
            CvInvoke.CvtColor(image3, grayImage, ColorConversion.Bgr2Gray);

            // show gray image
            //CvInvoke.Imshow("hello2", grayImage);



            // new binary image mat
            Mat biImage = new Mat();

            // from gray to binary
            CvInvoke.Threshold(grayImage, biImage, 128, 255, ThresholdType.Otsu);

            // show binary image
            //CvInvoke.Imshow("hello3", biImage);



            // Find contours
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();

            CvInvoke.FindContours(biImage, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Draw contours
            Mat contourImage = new Mat(biImage.Size, DepthType.Cv8U, 3);

            CvInvoke.DrawContours(contourImage, contours, -1, new MCvScalar(255, 0, 0));
            //CvInvoke.Imshow("Contours", contourImage);






            // contours to save
            VectorOfVectorOfPoint squareContours = new VectorOfVectorOfPoint();

            // loop through the found contours and filter them
            for (int i = 0; i < contours.Size; i++)
            {
                // input
                VectorOfPoint contour = contours[i];

                // for every contour, reduce the amount/number of point (/Approximate the contour) with Douglas-Peucker
                double epsilon = 4;
                bool closed = true;
                // output
                VectorOfPoint approxCurve = new VectorOfPoint();

                CvInvoke.ApproxPolyDP(contour, approxCurve, epsilon, closed);

                // save contours of .Size == 4. Discard all others.
                if (approxCurve.Size == 4)
                {
                    squareContours.Push(approxCurve);
                }


            }

            // Draw and show new squareContours drawn on image
            CvInvoke.DrawContours(image3, squareContours, -1, new MCvScalar(255, 0, 0));
            CvInvoke.Imshow("Contours2", image3);





            Matrix<float> rotationVector = new Matrix<float>(3, 1);
            Matrix<float> translationVector = new Matrix<float>(3, 1);
            Matrix<float> newRotMatrix = new Matrix<float>(3, 3);


            // Undistort and transform each figur in the bigger image, into seperat small images
            for (int i = 0; i < squareContours.Size; i++)
            {
                // input
                VectorOfPoint squaredContours = squareContours[i];
                // output
                VectorOfPointF newSquaredPoints = new VectorOfPointF();

                // new points for each contour
                newSquaredPoints.Push(new PointF[] { new PointF(0, 0), new PointF(100, 0), 
                    new PointF(100, 100), new PointF(0, 100) });

                // transform the squared contours using FindHomography
                Mat homography = CvInvoke.FindHomography(squaredContours, newSquaredPoints, RobustEstimationAlgorithm.Ransac);

                // create a new vector to hold the transformed points
                Mat transformedImage = new Mat();

                // warp the image using the homography matrix
                CvInvoke.WarpPerspective(image3, transformedImage, homography, new Size(100, 100));
                //CvInvoke.Imshow("bla" + i, transformedImage);
                Mat grayTransImage = new Mat();
                // make it gray
                CvInvoke.CvtColor(transformedImage, grayTransImage, ColorConversion.Bgr2Gray);

                Mat biTransImage = new Mat();
                // make binary
                CvInvoke.Threshold(grayTransImage, biTransImage, 128, 255, ThresholdType.Otsu);
                // show ALL binary transformed image
                CvInvoke.Imshow("Binary Transformed Image" + i, biTransImage);

                int numRows = 6;
                int numCols = 6;
                int cellSize = 100 / 6;

                // Calculate the center of each cell and get the pixel value of each cell (black or white)
                byte[,] pixelValues = new byte[numRows, numCols];
                for (int k = 0; k < numRows; k++)
                {
                    for (int l = 0; l < numCols; l++)
                    {
                        int x = (l * cellSize) + (cellSize / 2);
                        int y = (k * cellSize) + (cellSize / 2);
                        pixelValues[k, l] = biTransImage.GetRawData(new[] { x, y })[0];
                    }
                }
                

                // new matrix that takes in the pixelValues
                Matrix<byte> pMatrix = new Matrix<byte>(pixelValues);

                // Marker 1 normal
                Matrix<byte> Marker1 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker1 rotated 90 degrees clockwise
                Matrix<byte> Marker1Rot90 = new Matrix<byte>(new byte[100 / 6]);
                CvInvoke.Rotate(Marker1, Marker1Rot90, RotateFlags.Rotate90Clockwise);

                // Marker1 rotated 180 degrees
                Matrix<byte> Marker1Rot180 = new Matrix<byte>(new byte[100 / 6]);
                CvInvoke.Rotate(Marker1, Marker1Rot180, RotateFlags.Rotate180);

                // Marker1 rotated 270 degress clockwise
                Matrix<byte> Marker1Rot270 = new Matrix<byte>(new byte[100 / 6]);
                CvInvoke.Rotate(Marker1, Marker1Rot270, RotateFlags.Rotate90CounterClockwise);

                // compare pixelValues with Marker1
                bool marker1Equal = pMatrix.Equals(Marker1);
                // compare pixelValues with Marker1Rot90
                bool marker1Rot90Equal = pMatrix.Equals(Marker1Rot90);
                // compare pixelValues with Marker1Rot180
                bool marker1Rot180Equal = pMatrix.Equals(Marker1Rot180);
                // compare pixelValues with Marker1Rot270
                bool marker1Rot270Equal = pMatrix.Equals(Marker1Rot270);


                //// Marker 2 normal
                //Matrix<byte> marker2 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker2
                //bool marker2Equal = pMatrix.Equals(marker2);

                //// Marker 3 normal
                //Matrix<byte> marker3 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker3
                //bool marker3Equal = pMatrix.Equals(marker3);

                //// Marker 4 normal
                //Matrix<byte> marker4 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0, 255, 255,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker4
                //bool marker4Equal = pMatrix.Equals(marker4);

                //// Marker 5 normal
                //Matrix<byte> marker5 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker5
                //bool marker5Equal = pMatrix.Equals(marker5);

                //// Marker 6 normal
                //Matrix<byte> marker6 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker6
                //bool marker6Equal = pMatrix.Equals(marker6);

                //// Marker 7 normal
                //Matrix<byte> marker7 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255,   0,   0, 0 },
                //    { 0, 255, 255, 255,   0, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker7
                //bool marker7Equal = pMatrix.Equals(marker7);

                //// Marker 8 normal
                //Matrix<byte> marker8 = new Matrix<byte>(new byte[,]
                //{
                //    { 0,   0,   0,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255,   0,   0, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0, 255, 255, 255, 255, 0 },
                //    { 0,   0,   0,   0,   0, 0 }
                //});
                //// compare pixelValues with Marker8
                //bool marker8Equal = pMatrix.Equals(marker8);


                // Convert VectorOfPointF points to MCvPoint3D32f
                MCvPoint3D32f[] mcPoints = new MCvPoint3D32f[newSquaredPoints.Size];
                for (int n = 0; n < newSquaredPoints.Size; n++)
                {
                    PointF point = newSquaredPoints[n];
                    mcPoints[n] = new MCvPoint3D32f(point.X, point.Y, 0);
                }


                //MCvPoint3D32f[] objectPointsMarker1 = new MCvPoint3D32f[] {
                //        new MCvPoint3D32f(0, 0, 0),
                //        new MCvPoint3D32f(300, 0, 0),
                //        new MCvPoint3D32f(300, 300, 0),
                //        new MCvPoint3D32f(0, 300, 0)
                //};

                // Define the image points
                Point[] points = squareContours[i].ToArray();
                PointF[] imagePoints = points.Select(p => new PointF(p.X, p.Y)).ToArray();

                // Estimate the pose using SolvePnP
                

                CvInvoke.SolvePnP(mcPoints, imagePoints, intrinsic, distortionCoeff, rotationVector, translationVector);

                

                CvInvoke.Rodrigues(rotationVector, newRotMatrix);


                // New matrix from our new rotaion matrix's data and translation data
                float[,] rValues = newRotMatrix.Data;
                float[,] tValues = translationVector.Data;

                Matrix<float> rtMatrix = new Matrix<float>(new float[,]
                {
                        { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
                        { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
                        { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
                });
                UtilityAR.DrawCube(image3, intrinsic * rtMatrix);
                CvInvoke.Imshow("draw cube", image3);



                if (marker1Equal)
                {
                    Console.WriteLine("Marker1 and pMatrix are equal");
                }
                else
                {
                    Console.WriteLine("Marker1 and pMatrix are NOT equal");
                }

                //if (marker2Equal)
                //{
                //    Console.WriteLine("Marker2 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker2 and pMatrix are NOT equal");
                //}
                //if (marker3Equal)
                //{
                //    Console.WriteLine("Marker3 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker3 and pMatrix are NOT equal");
                //}
                //if (marker4Equal)
                //{
                //    Console.WriteLine("Marker4 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker4 and pMatrix are NOT equal");
                //}
                //if (marker5Equal)
                //{
                //    Console.WriteLine("Marker5 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker5 and pMatrix are NOT equal");
                //}
                //if (marker6Equal)
                //{
                //    Console.WriteLine("Marker6 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker6 and pMatrix are NOT equal");
                //}
                //if (marker7Equal)
                //{
                //    Console.WriteLine("Marker7 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker7 and pMatrix are NOT equal");
                //}
                //if (marker8Equal)
                //{
                //    Console.WriteLine("Marker8 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker8 and pMatrix are NOT equal");
                //}

                //if (marker1Rot90Equal)
                //{
                //    Console.WriteLine("Marker1Rot90 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker1Rot90 and pMatrix are NOT equal");
                //}

                //if (marker1Rot180Equal)
                //{
                //    Console.WriteLine("Marker1Rot180 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker1Rot180 and pMatrix are NOT equal");
                //}

                //if (marker1Rot270Equal)
                //{
                //    Console.WriteLine("Marker1Rot270 and pMatrix are equal");
                //}
                //else
                //{
                //    Console.WriteLine("Marker1Rot270 and pMatrix are NOT equal");
                //}

                if (marker1Equal)
                {
                    Console.WriteLine("Marker1 and pMatrix are equal");
                }
                else
                {
                    Console.WriteLine("Marker1 and pMatrix are NOT equal");
                }

                string attackValue = "4";
                MCvScalar redColor = new MCvScalar(0, 0, 255);


                //UtilityAR.DrawRedCube(image3, intrinsic * rtMatrix, attackValue, redColor);

                

                
                

                //if (marker1Equal)
                //{
                //    Console.WriteLine("yes");
                //}
                //else
                //{
                //    Console.WriteLine("no");
                //}


                //if (marker1Equal || marker1Rot90Equal || marker1Rot180Equal || marker1Rot270Equal)
                //{

                //    //UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                //    //CvInvoke.Imshow("draw cube", image);
                //}

            }
            





        }



        



        public override void OnFrame()
        {
            //Mat video = new Mat();

            //vCap.Read(video);


            ////Mat image2 = CvInvoke.Imread("capture_28.jpg");
            //Mat image3 = CvInvoke.Imread("capture_0.jpg");

            //Mat grayImage2 = new Mat();
            //CvInvoke.CvtColor(video, grayImage2, ColorConversion.Bgr2Gray);

            //Mat biImage2 = new Mat();
            //CvInvoke.Threshold(grayImage2, biImage2, 128, 255, ThresholdType.Otsu);

            //VectorOfVectorOfPoint contours2 = new VectorOfVectorOfPoint();
            //Mat hierarchy2 = new Mat();

            //CvInvoke.FindContours(biImage2, contours2, hierarchy2, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            //// Draw contours
            //Mat contourImage2 = new Mat(biImage2.Size, DepthType.Cv8U, 3);

            //CvInvoke.DrawContours(contourImage2, contours2, -1, new MCvScalar(255, 0, 0));

            //VectorOfVectorOfPoint squareContours2 = new VectorOfVectorOfPoint();

            //// loop through the found contours and filter them
            //for (int i = 0; i < contours2.Size; i++)
            //{
            //    // input
            //    VectorOfPoint contour2 = contours2[i];

            //    // for every contour, reduce the amount/number of point (/Approximate the contour) with Douglas-Peucker
            //    double epsilon = 4;
            //    bool closed = true;
            //    // output
            //    VectorOfPoint approxCurve = new VectorOfPoint();

            //    CvInvoke.ApproxPolyDP(contour2, approxCurve, epsilon, closed);

            //    // save contours of .Size == 4. Discard all others.
            //    if (approxCurve.Size == 4)
            //    {
            //        squareContours2.Push(approxCurve);
            //    }


            //}

            //// Draw and show new squareContours drawn on image
            //CvInvoke.DrawContours(video, squareContours2, -1, new MCvScalar(255, 0, 0));
            ////CvInvoke.Imshow("video contours", video);

            //for (int i = 0; i < squareContours2.Size; i++)
            //{
            //    // input
            //    VectorOfPoint squaredContours2 = squareContours2[i];
            //    // output
            //    VectorOfPointF newSquaredPoints2 = new VectorOfPointF();

            //    // new points for each contour
            //    newSquaredPoints2.Push(new PointF[] { new PointF(0, 0), new PointF(1, 0), new PointF(1, 1), new PointF(0, 1) });

            //    // transform the squared contours using FindHomography
            //    Mat homography2 = CvInvoke.FindHomography(squaredContours2, newSquaredPoints2, RobustEstimationAlgorithm.Ransac);

            //    // create a new vector to hold the transformed points
            //    Mat transformedImage2 = new Mat();

            //    // warp the image using the homography matrix
            //    CvInvoke.WarpPerspective(video, transformedImage2, homography2, new Size(100, 100));

            //    Mat grayTransImage2 = new Mat();
            //    // make it gray
            //    CvInvoke.CvtColor(transformedImage2, grayTransImage2, ColorConversion.Bgr2Gray);

            //    Mat biTransImage2 = new Mat();
            //    // make binary
            //    CvInvoke.Threshold(grayTransImage2, biTransImage2, 128, 255, ThresholdType.Otsu);
            //    //CvInvoke.Imshow("Binary Transformed Image" + i, biTransImage2);


            //    int numRows = 6;
            //    int numCols = 6;
            //    int cellSize = 6 / 6;

            //    // Calculate the center of each cell and get the pixel value of each cell (black or white)
            //    byte[,] pixelValues2 = new byte[numRows, numCols];
            //    for (int k = 0; k < numRows; k++)
            //    {
            //        for (int l = 0; l < numCols; l++)
            //        {
            //            int x = (l * cellSize) + (cellSize / 2);
            //            int y = (k * cellSize) + (cellSize / 2);
            //            pixelValues2[k, l] = biTransImage2.GetRawData(new[] { x, y })[0];
            //        }
            //    }

            //    // new matrix that takes in the pixelValues
            //    Matrix<byte> pMatrix2 = new Matrix<byte>(pixelValues2);
            //    // Marker 1 normal
            //    Matrix<byte> Marker2 = new Matrix<byte>(new byte[,]
            //    {
            //        { 0,   0,   0,   0,   0, 0 },
            //        { 0, 255, 255, 255, 255, 0 },
            //        { 0, 255, 255, 255, 255, 0 },
            //        { 0, 255, 255,   0,   0, 0 },
            //        { 0, 255, 255, 255, 255, 0 },
            //        { 0,   0,   0,   0,   0, 0 }
            //    });

            //    // Marker1 rotated 90 degrees clockwise
            //    Matrix<byte> Marker2Rot90 = new Matrix<byte>(new byte[6 / 6]);
            //    CvInvoke.Rotate(Marker2, Marker2Rot90, RotateFlags.Rotate90Clockwise);

            //    // Marker1 rotated 180 degrees
            //    Matrix<byte> Marker2Rot180 = new Matrix<byte>(new byte[6 / 6]);
            //    CvInvoke.Rotate(Marker2, Marker2Rot180, RotateFlags.Rotate180);

            //    // Marker1 rotated 270 degress clockwise
            //    Matrix<byte> Marker2Rot270 = new Matrix<byte>(new byte[6 / 6]);
            //    CvInvoke.Rotate(Marker2, Marker2Rot270, RotateFlags.Rotate90CounterClockwise);

            //    // compare pixelValues with Marker1
            //    bool marker2Equal = pMatrix2.Equals(Marker2);
            //    // compare pixelValues with Marker1Rot90
            //    bool marker2Rot90Equal = pMatrix2.Equals(Marker2Rot90);
            //    // compare pixelValues with Marker1Rot180
            //    bool marker2Rot180Equal = pMatrix2.Equals(Marker2Rot180);
            //    // compare pixelValues with Marker1Rot270
            //    bool marker2Rot270Equal = pMatrix2.Equals(Marker2Rot270);

                

            //    // Convert VectorOfPointF points to MCvPoint3D32f
            //    MCvPoint3D32f[] mcPoints2 = new MCvPoint3D32f[newSquaredPoints2.Size];
            //    for (int n = 0; n < newSquaredPoints2.Size; n++)
            //    {
            //        PointF point = newSquaredPoints2[n];
            //        mcPoints2[n] = new MCvPoint3D32f(point.X, point.Y, 0);
            //    }

            //    // Define the image points
            //    Point[] points = squareContours2[i].ToArray();
            //    PointF[] imagePoints2 = points.Select(p => new PointF(p.X, p.Y)).ToArray();

            //    // Estimate the pose using SolvePnP
                

            //    CvInvoke.SolvePnP(mcPoints2, imagePoints2, intrinsic, distortionCoeff, rotationVector2, translationVector2);

                

            //    CvInvoke.Rodrigues(rotationVector2, newRotMatrix2);

            //    // New matrix from our new rotaion matrix's data and translation data
            //    float[,] rValues = newRotMatrix2.Data;
            //    float[,] tValues = translationVector2.Data;

            //    Matrix<float> rtMatrix2 = new Matrix<float>(new float[,]
            //    {
            //            { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
            //            { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
            //            { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
            //    });

            //    string attackValue = "6";
            //    MCvScalar greenColor = new MCvScalar(0, 255, 0);

            //    if (marker2Equal)
            //    {
            //        Console.WriteLine("Equal");
            //    }
            //    else
            //    {
            //        Console.WriteLine("NOT equal");
            //    }

            //    //if (marker2Rot90Equal)
            //    //{
            //    //    Console.WriteLine("Marker1Rot90 and pMatrix are equal");
            //    //}
            //    //else
            //    //{
            //    //    Console.WriteLine("Marker1Rot90 and pMatrix are NOT equal");
            //    //}

            //    //if (marker2Rot180Equal)
            //    //{
            //    //    Console.WriteLine("Marker1Rot180 and pMatrix are equal");
            //    //}
            //    //else
            //    //{
            //    //    Console.WriteLine("Marker1Rot180 and pMatrix are NOT equal");
            //    //}

            //    //if (marker2Rot270Equal)
            //    //{
            //    //    Console.WriteLine("Marker1Rot270 and pMatrix are equal");
            //    //}
            //    //else
            //    //{
            //    //    Console.WriteLine("Marker1Rot270 and pMatrix are NOT equal");
            //    //}

            //    //UtilityAR.DrawCube(video, intrinsic * rtMatrix2);


            //    //CvInvoke.Imshow("draw cube", video);

            //    //foreach (var item in attackValue)
            //    //{

            //    //}
            //}



            //CvInvoke.Imshow("Video", video);
        }
    }
}
