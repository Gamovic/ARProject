﻿using System;
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

        Matrix<float> rotationVector = new Matrix<float>(3, 1);
        Matrix<float> translationVector = new Matrix<float>(3, 1);
        Matrix<float> newRotMatrix = new Matrix<float>(3, 3);

        Matrix<float> rtMatrix;

        bool marker1Equal, marker1Rot90Equal, marker1Rot180Equal, marker1Rot270Equal;
        bool marker2Equal, marker2Rot90Equal, marker2Rot180Equal, marker2Rot270Equal;
        bool marker3Equal, marker3Rot90Equal, marker3Rot180Equal, marker3Rot270Equal;

        public MarkerAR()
        {
            vCap = new VideoCapture(1);

            intrinsic = new Matrix<float>(3, 3);
            distortionCoeff = new Matrix<float>(1, 5);
            //Read intrinsic and distortionCoeff from CameraCalibration (.json file)
            UtilityAR.ReadIntrinsicsFromFile(out intrinsic, out distortionCoeff);

            // Keep on capturing images
            //UtilityAR.CaptureLoop(new Size(7, 4), 1);
            //UtilityAR.CalibrateCamera(new Size(7, 4), true);

            // load image
            Mat image = CvInvoke.Imread("capture_1.jpg");

            // new gray image mat
            Mat grayImage = new Mat();

            // from color to gray
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

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
            CvInvoke.DrawContours(image, squareContours, -1, new MCvScalar(255, 0, 0));
            CvInvoke.Imshow("Contours2", image);


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
                CvInvoke.WarpPerspective(image, transformedImage, homography, new Size(100, 100));
                //CvInvoke.Imshow("bla" + i, transformedImage);
                Mat grayTransImage = new Mat();
                // make it gray
                CvInvoke.CvtColor(transformedImage, grayTransImage, ColorConversion.Bgr2Gray);

                Mat biTransImage = new Mat();
                // make binary
                CvInvoke.Threshold(grayTransImage, biTransImage, 128, 255, ThresholdType.Otsu);
                // show ALL binary transformed image
                //CvInvoke.Imshow("Binary Transformed Image" + i, biTransImage);

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
                Matrix<byte> pixelMatrix = new Matrix<byte>(pixelValues);

                ///
                /// Marker 1 - Works with: "capture_1.jpg"
                ///
                Matrix<byte> marker1 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 90 degrees clockwise - Works with: "capture_16.jpg"
                Matrix<byte> marker1Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 180 degrees - Works with: "capture_11.jpg"
                Matrix<byte> marker1Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> marker1Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker1
                marker1Equal = pixelMatrix.Equals(marker1);
                marker1Rot90Equal = pixelMatrix.Equals(marker1Rot90);
                marker1Rot180Equal = pixelMatrix.Equals(marker1Rot180);
                marker1Rot270Equal = pixelMatrix.Equals(marker1Rot270);

                ///
                /// Marker 2 - Works with: "capture_1.jpg"
                ///
                Matrix<byte> marker2 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 2 rotated 90 degrees clockwise - Works with: "capture_16.jpg"
                Matrix<byte> marker2Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                //Matrix<byte> Marker2Rot90 = new Matrix<byte>(new byte[100 / 6]);
                //CvInvoke.Rotate(Marker2, Marker2Rot90, RotateFlags.Rotate90Clockwise);

                // Marker 2 rotated 180 degrees - Works with: "capture_11.jpg"
                Matrix<byte> marker2Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                //Matrix<byte> Marker2Rot180 = new Matrix<byte>(new byte[100 / 6]);
                //CvInvoke.Rotate(Marker2, Marker2Rot180, RotateFlags.Rotate180);

                // Marker 2 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> marker2Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                //Matrix<byte> Marker2Rot270 = new Matrix<byte>(new byte[100 / 6]);
                //CvInvoke.Rotate(Marker2, Marker2Rot270, RotateFlags.Rotate90CounterClockwise);

                // compare pixelValues with Marker2
                marker2Equal = pixelMatrix.Equals(marker2);
                // compare pixelValues with Marker2Rot90
                marker2Rot90Equal = pixelMatrix.Equals(marker2Rot90);
                // compare pixelValues with Marker2Rot180
                marker2Rot180Equal = pixelMatrix.Equals(marker2Rot180);
                // compare pixelValues with Marker2Rot270
                marker2Rot270Equal = pixelMatrix.Equals(marker2Rot270);

                ///
                /// Marker 3 normal - Works with: "capture_1.jpg"
                ///
                Matrix<byte> Marker3 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 90 degrees clockwise - Works with: "capture_16.jpg"
                Matrix<byte> Marker3Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 180 degrees - Works with: "capture_11.jpg"
                Matrix<byte> Marker3Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> Marker3Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // compare pixelValues with Marker3
                marker3Equal = pixelMatrix.Equals(Marker3);
                // compare pixelValues with Marker3Rot90
                marker3Rot90Equal = pixelMatrix.Equals(Marker3Rot90);
                // compare pixelValues with Marker3Rot180
                marker3Rot180Equal = pixelMatrix.Equals(Marker3Rot180);
                // compare pixelValues with Marker3Rot270
                marker3Rot270Equal = pixelMatrix.Equals(Marker3Rot270);

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

                // Define the image points
                Point[] points = squareContours[i].ToArray();
                PointF[] imagePoints = points.Select(p => new PointF(p.X, p.Y)).ToArray();

                // Estimate the pose using SolvePnP
                

                CvInvoke.SolvePnP(mcPoints, imagePoints, intrinsic, distortionCoeff, rotationVector, translationVector);

                

                CvInvoke.Rodrigues(rotationVector, newRotMatrix);


                // New matrix from our new rotaion matrix's data and translation data
                float[,] rValues = newRotMatrix.Data;
                float[,] tValues = translationVector.Data;

                rtMatrix = new Matrix<float>(new float[,]
                {
                        { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
                        { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
                        { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
                });

                ///
                /// Check marker 1 and draw cube if pixelMatrix equals marker 1
                ///
                if (marker1Equal)
                {
                    Console.WriteLine("Marker1 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);

                }
                if (marker1Rot90Equal)
                {
                    Console.WriteLine("Marker1Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);

                }
                if (marker1Rot180Equal)
                {
                    Console.WriteLine("Marker1Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);

                }
                if (marker1Rot270Equal)
                {
                    Console.WriteLine("Marker1Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);

                }

                ///
                /// Check marker 2 and draw cube if pixelMatrix equals marker 2
                ///
                if (marker2Equal)
                {
                    Console.WriteLine("Marker2 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker2Rot90Equal)
                {
                    Console.WriteLine("Marker2Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker2Rot180Equal)
                {
                    Console.WriteLine("Marker2Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker2Rot270Equal)
                {
                    Console.WriteLine("Marker2Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }

                ///
                /// Check marker 3 and draw cube if pixelMatrix equals marker 3
                ///
                if (marker3Equal)
                {
                    Console.WriteLine("Marker3 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker3Rot90Equal)
                {
                    Console.WriteLine("Marker3Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker3Rot180Equal)
                {
                    Console.WriteLine("Marker3Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker3Rot270Equal)
                {
                    Console.WriteLine("Marker3Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }



                //if (marker4Equal)
                //{
                //    Console.WriteLine("Marker4 and pMatrix are equal");
                //}
                //if (marker5Equal)
                //{
                //    Console.WriteLine("Marker5 and pMatrix are equal");
                //}
                //if (marker6Equal)
                //{
                //    Console.WriteLine("Marker6 and pMatrix are equal");
                //}
                //if (marker7Equal)
                //{
                //    Console.WriteLine("Marker7 and pMatrix are equal");
                //}
                //if (marker8Equal)
                //{
                //    Console.WriteLine("Marker8 and pMatrix are equal");
                //}



                //UtilityAR.DrawRedCube(image3, intrinsic * rtMatrix, attackValue, redColor);

                //UtilityAR.DrawCube(image, intrinsic * rtMatrix);

                //CvInvoke.Imshow("draw cube", image);


            }


            //UtilityAR.DrawCube(image, intrinsic * rtMatrix);

            CvInvoke.Imshow("draw cube", image);
        }



        public override void OnFrame()
        {
            Mat video = new Mat();

            vCap.Read(video);

            Mat grayImage = new Mat();
            CvInvoke.CvtColor(video, grayImage, ColorConversion.Bgr2Gray);

            Mat biImage = new Mat();
            CvInvoke.Threshold(grayImage, biImage, 128, 255, ThresholdType.Otsu);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();

            CvInvoke.FindContours(biImage, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Draw contours
            Mat contourImage = new Mat(biImage.Size, DepthType.Cv8U, 3);

            CvInvoke.DrawContours(contourImage, contours, -1, new MCvScalar(255, 0, 0));

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
            CvInvoke.DrawContours(video, squareContours, -1, new MCvScalar(255, 0, 0));
            //CvInvoke.Imshow("video contours", video);

            for (int i = 0; i < squareContours.Size; i++)
            {
                // input
                VectorOfPoint squaredContours = squareContours[i];
                // output
                VectorOfPointF newSquaredPoints = new VectorOfPointF();

                // new points for each contour
                newSquaredPoints.Push(new PointF[] { new PointF(0, 0), new PointF(100, 0), new PointF(100, 100), new PointF(0, 100) });

                // transform the squared contours using FindHomography
                Mat homography = CvInvoke.FindHomography(squaredContours, newSquaredPoints, RobustEstimationAlgorithm.Ransac);

                // create a new vector to hold the transformed points
                Mat transformedImage = new Mat();

                // warp the image using the homography matrix
                CvInvoke.WarpPerspective(video, transformedImage, homography, new Size(100, 100));

                Mat grayTransImage = new Mat();
                // make it gray
                CvInvoke.CvtColor(transformedImage, grayTransImage, ColorConversion.Bgr2Gray);

                Mat biTransImage = new Mat();
                // make binary
                CvInvoke.Threshold(grayTransImage, biTransImage, 128, 255, ThresholdType.Otsu);
                //CvInvoke.Imshow("Binary Transformed Image" + i, biTransImage2);


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

                ///
                /// Marker 1
                /// 
                Matrix<byte> marker1 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 90 degrees clockwise
                Matrix<byte> marker1Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 180 degrees
                Matrix<byte> marker1Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 1 rotated 270 degress clockwise
                Matrix<byte> marker1Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker2
                marker1Equal = pMatrix.Equals(marker1);
                marker1Rot90Equal = pMatrix.Equals(marker1Rot90);
                marker1Rot180Equal = pMatrix.Equals(marker1Rot180);
                marker1Rot270Equal = pMatrix.Equals(marker1Rot270);

                ///
                /// Marker 2
                /// 
                Matrix<byte> marker2 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });


                // Marker 2 rotated 90 degrees clockwise
                Matrix<byte> marker2Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 2 rotated 180 degrees
                Matrix<byte> marker2Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 2 rotated 270 degress clockwise
                Matrix<byte> marker2Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // compare pixelValues with Marker1
                marker2Equal = pMatrix.Equals(marker2);
                // compare pixelValues with Marker1Rot90
                marker2Rot90Equal = pMatrix.Equals(marker2Rot90);
                // compare pixelValues with Marker1Rot180
                marker2Rot180Equal = pMatrix.Equals(marker2Rot180);
                // compare pixelValues with Marker1Rot270
                marker2Rot270Equal = pMatrix.Equals(marker2Rot270);
                
                ///
                /// Marker 3 normal
                /// 
                Matrix<byte> Marker3 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 90 degrees clockwise
                Matrix<byte> Marker3Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 180 degrees
                Matrix<byte> Marker3Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 3 rotated 270 degress clockwise
                Matrix<byte> Marker3Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // compare pixelValues with Marker3
                marker3Equal = pMatrix.Equals(Marker3);
                // compare pixelValues with Marker3Rot90
                marker3Rot90Equal = pMatrix.Equals(Marker3Rot90);
                // compare pixelValues with Marker3Rot180
                marker3Rot180Equal = pMatrix.Equals(Marker3Rot180);
                // compare pixelValues with Marker3Rot270
                marker3Rot270Equal = pMatrix.Equals(Marker3Rot270);


                // Convert VectorOfPointF points to MCvPoint3D32f
                MCvPoint3D32f[] mcPoints = new MCvPoint3D32f[newSquaredPoints.Size];
                for (int n = 0; n < newSquaredPoints.Size; n++)
                {
                    PointF point = newSquaredPoints[n];
                    mcPoints[n] = new MCvPoint3D32f(point.X, point.Y, 0);
                }

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

                string cube1 = "1";
                string cube2 = "2";
                string cube3 = "3";
                MCvScalar greenColor = new MCvScalar(0, 255, 0);
                MCvScalar blueColor = new MCvScalar(255, 0, 0);
                MCvScalar yellowColor = new MCvScalar(0, 255, 255);
                MCvScalar redColor = new MCvScalar(0, 0, 255);

                if (marker1Equal || marker1Rot90Equal || marker1Rot180Equal || marker1Rot270Equal)
                {
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, cube1, greenColor);
                }

                if (marker2Equal || marker2Rot90Equal || marker2Rot180Equal || marker2Rot270Equal)
                {
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, cube2, blueColor);
                }

                if (marker3Equal || marker3Rot90Equal || marker3Rot180Equal || marker3Rot270Equal)
                {
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, cube3, yellowColor);
                }


                //CvInvoke.Imshow("draw cube", video);

            }



            CvInvoke.Imshow("Video", video);
        }
    }
}
