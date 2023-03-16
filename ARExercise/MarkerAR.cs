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

        Mat image;
        Mat video;
        Mat grayImage;
        Mat binaryImage;
        Mat hierarchy;
        Mat contourImage;

        Matrix<float> intrinsic;
        Matrix<float> distortionCoeff;

        Matrix<float> rotationVector = new Matrix<float>(3, 1);
        Matrix<float> translationVector = new Matrix<float>(3, 1);
        Matrix<float> newRotMatrix = new Matrix<float>(3, 3);

        Matrix<byte> pixelMatrix;

        Matrix<float> rtMatrix;

        VectorOfVectorOfPoint contours;

        VectorOfVectorOfPoint squareContours;

        bool marker1Equal, marker1Rot90Equal, marker1Rot180Equal, marker1Rot270Equal;
        bool marker2Equal, marker2Rot90Equal, marker2Rot180Equal, marker2Rot270Equal;
        bool marker3Equal, marker3Rot90Equal, marker3Rot180Equal, marker3Rot270Equal;
        bool marker4Equal, marker4Rot90Equal, marker4Rot180Equal, marker4Rot270Equal;
        bool marker5Equal, marker5Rot90Equal, marker5Rot180Equal, marker5Rot270Equal;
        bool marker6Equal, marker6Rot90Equal, marker6Rot180Equal, marker6Rot270Equal;
        bool marker7Equal, marker7Rot90Equal, marker7Rot180Equal, marker7Rot270Equal;
        bool marker8Equal, marker8Rot90Equal, marker8Rot180Equal, marker8Rot270Equal;

        public MarkerAR()
        {
            vCap = new VideoCapture(1);

            intrinsic = new Matrix<float>(3, 3);
            distortionCoeff = new Matrix<float>(1, 5);
            //Read intrinsic and distortionCoeff from CameraCalibration (.json file)
            UtilityAR.ReadIntrinsicsFromFile(out intrinsic, out distortionCoeff);

            // load image
            image = CvInvoke.Imread("capture_1.jpg");
            // new gray image mat
            grayImage = new Mat();
            // new binary image mat
            binaryImage = new Mat();

            hierarchy = new Mat();

            contours = new VectorOfVectorOfPoint();
            

            // from color to gray
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

            // from gray to binary
            CvInvoke.Threshold(grayImage, binaryImage, 128, 255, ThresholdType.Otsu);

            // Find contours
            CvInvoke.FindContours(binaryImage, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Draw contours
            contourImage = new Mat(binaryImage.Size, DepthType.Cv8U, 3);
            CvInvoke.DrawContours(contourImage, contours, -1, new MCvScalar(255, 0, 0));
            //CvInvoke.Imshow("Contours", contourImage);

            // contours to save
            squareContours = new VectorOfVectorOfPoint();

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
                pixelMatrix = new Matrix<byte>(pixelValues);

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
                // compare pixelValues with Marker 1
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

                // compare pixelValues with Marker 2
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
                // Marker 3 rotated 90 degrees clockwise - Works with: "capture_2.jpg"
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

                // compare pixelValues with Marker 3
                marker3Equal = pixelMatrix.Equals(Marker3);
                // compare pixelValues with Marker3Rot90
                marker3Rot90Equal = pixelMatrix.Equals(Marker3Rot90);
                // compare pixelValues with Marker3Rot180
                marker3Rot180Equal = pixelMatrix.Equals(Marker3Rot180);
                // compare pixelValues with Marker3Rot270
                marker3Rot270Equal = pixelMatrix.Equals(Marker3Rot270);

                ///
                /// Marker 4 normal - Works with: video
                ///
                Matrix<byte> Marker4 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 90 degrees clockwise - Works with: video
                Matrix<byte> Marker4Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 180 degrees - Works with: video
                Matrix<byte> Marker4Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 270 degress clockwise - Works with: video
                Matrix<byte> Marker4Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // compare pixelValues with Marker 4
                marker4Equal = pixelMatrix.Equals(Marker4);
                // compare pixelValues with Marker3Rot90
                marker4Rot90Equal = pixelMatrix.Equals(Marker4Rot90);
                // compare pixelValues with Marker3Rot180
                marker4Rot180Equal = pixelMatrix.Equals(Marker4Rot180);
                // compare pixelValues with Marker3Rot270
                marker4Rot270Equal = pixelMatrix.Equals(Marker4Rot270);

                ///
                /// Marker 5 - Works with: video
                ///
                Matrix<byte> marker5 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 5 rotated 90 degrees clockwise - Works with: video
                Matrix<byte> marker5Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 5 rotated 180 degrees - Works with: video
                Matrix<byte> marker5Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // Marker 5 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> marker5Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });

                // compare pixelValues with Marker 5
                marker5Equal = pixelMatrix.Equals(marker5);
                // compare pixelValues with Marker2Rot90
                marker5Rot90Equal = pixelMatrix.Equals(marker5Rot90);
                // compare pixelValues with Marker2Rot180
                marker5Rot180Equal = pixelMatrix.Equals(marker5Rot180);
                // compare pixelValues with Marker2Rot270
                marker5Rot270Equal = pixelMatrix.Equals(marker5Rot270);

                ///
                /// Marker 6 - Works with: video
                ///
                Matrix<byte> marker6 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 90 degrees clockwise - Works with: video
                Matrix<byte> marker6Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 180 degrees - Works with: video
                Matrix<byte> marker6Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 270 degress clockwise - Works with: video
                Matrix<byte> marker6Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 6
                marker6Equal = pixelMatrix.Equals(marker6);
                marker6Rot90Equal = pixelMatrix.Equals(marker6Rot90);
                marker6Rot180Equal = pixelMatrix.Equals(marker6Rot180);
                marker6Rot270Equal = pixelMatrix.Equals(marker6Rot270);

                ///
                /// Marker 7 - Works with: "capture_1.jpg"
                ///
                Matrix<byte> marker7 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 90 degrees clockwise - Works with: "capture_16.jpg"
                Matrix<byte> marker7Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 180 degrees - Works with: "capture_11.jpg"
                Matrix<byte> marker7Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> marker7Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 7
                marker7Equal = pixelMatrix.Equals(marker7);
                marker7Rot90Equal = pixelMatrix.Equals(marker7Rot90);
                marker7Rot180Equal = pixelMatrix.Equals(marker7Rot180);
                marker7Rot270Equal = pixelMatrix.Equals(marker7Rot270);

                ///
                /// Marker 8 - Works with: "capture_1.jpg"
                ///
                Matrix<byte> marker8 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 90 degrees clockwise - Works with: "capture_16.jpg"
                Matrix<byte> marker8Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 180 degrees - Works with: "capture_11.jpg"
                Matrix<byte> marker8Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 270 degress clockwise - Works with: "capture_6.jpg"
                Matrix<byte> marker8Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 8
                marker8Equal = pixelMatrix.Equals(marker8);
                marker8Rot90Equal = pixelMatrix.Equals(marker8Rot90);
                marker8Rot180Equal = pixelMatrix.Equals(marker8Rot180);
                marker8Rot270Equal = pixelMatrix.Equals(marker8Rot270);

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

                ///
                /// Check marker 4 and draw cube if pixelMatrix equals marker 4
                ///
                if (marker4Equal)
                {
                    Console.WriteLine("Marker4 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker4Rot90Equal)
                {
                    Console.WriteLine("Marker4Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker4Rot180Equal)
                {
                    Console.WriteLine("Marker4Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker4Rot270Equal)
                {
                    Console.WriteLine("Marker4Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }

                ///
                /// Check marker 7 and draw cube if pixelMatrix equals marker 7
                ///
                if (marker7Equal)
                {
                    Console.WriteLine("Marker7 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker7Rot90Equal)
                {
                    Console.WriteLine("Marker7Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker7Rot180Equal)
                {
                    Console.WriteLine("Marker7Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker7Rot270Equal)
                {
                    Console.WriteLine("Marker7Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }

                ///
                /// Check marker 8 and draw cube if pixelMatrix equals marker 8
                ///
                if (marker8Equal)
                {
                    Console.WriteLine("Marker8 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker8Rot90Equal)
                {
                    Console.WriteLine("Marker8Rot90 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker8Rot180Equal)
                {
                    Console.WriteLine("Marker8Rot180 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
                if (marker8Rot270Equal)
                {
                    Console.WriteLine("Marker8Rot270 and pMatrix are equal");
                    UtilityAR.DrawCube(image, intrinsic * rtMatrix);
                }
            }
            CvInvoke.Imshow("draw cube", image);
        }



        public override void OnFrame()
        {
            video = new Mat();

            vCap.Read(video);

            CvInvoke.CvtColor(video, grayImage, ColorConversion.Bgr2Gray);

            CvInvoke.Threshold(grayImage, binaryImage, 128, 255, ThresholdType.Otsu);

            CvInvoke.FindContours(binaryImage, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Draw contours
            CvInvoke.DrawContours(contourImage, contours, -1, new MCvScalar(255, 0, 0));

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

                Mat binaryTransformedImage = new Mat();
                // make binary
                CvInvoke.Threshold(grayTransImage, binaryTransformedImage, 128, 255, ThresholdType.Otsu);

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
                        pixelValues[k, l] = binaryTransformedImage.GetRawData(new[] { x, y })[0];
                    }
                }

                // new matrix that takes in the pixelValues
                Matrix<byte> pixelMatrix = new Matrix<byte>(pixelValues);

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
                marker1Equal = pixelMatrix.Equals(marker1);
                marker1Rot90Equal = pixelMatrix.Equals(marker1Rot90);
                marker1Rot180Equal = pixelMatrix.Equals(marker1Rot180);
                marker1Rot270Equal = pixelMatrix.Equals(marker1Rot270);

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
                marker2Equal = pixelMatrix.Equals(marker2);
                // compare pixelValues with Marker1Rot90
                marker2Rot90Equal = pixelMatrix.Equals(marker2Rot90);
                // compare pixelValues with Marker1Rot180
                marker2Rot180Equal = pixelMatrix.Equals(marker2Rot180);
                // compare pixelValues with Marker1Rot270
                marker2Rot270Equal = pixelMatrix.Equals(marker2Rot270);
                
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
                marker3Equal = pixelMatrix.Equals(Marker3);
                // compare pixelValues with Marker3Rot90
                marker3Rot90Equal = pixelMatrix.Equals(Marker3Rot90);
                // compare pixelValues with Marker3Rot180
                marker3Rot180Equal = pixelMatrix.Equals(Marker3Rot180);
                // compare pixelValues with Marker3Rot270
                marker3Rot270Equal = pixelMatrix.Equals(Marker3Rot270);

                ///
                /// Marker 4
                ///
                Matrix<byte> Marker4 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 90 degrees clockwise
                Matrix<byte> Marker4Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 180 degrees
                Matrix<byte> Marker4Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 4 rotated 270 degress clockwise
                Matrix<byte> Marker4Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker3
                marker4Equal = pixelMatrix.Equals(Marker4);
                // compare pixelValues with Marker3Rot90
                marker4Rot90Equal = pixelMatrix.Equals(Marker4Rot90);
                // compare pixelValues with Marker3Rot180
                marker4Rot180Equal = pixelMatrix.Equals(Marker4Rot180);
                // compare pixelValues with Marker3Rot270
                marker4Rot270Equal = pixelMatrix.Equals(Marker4Rot270);

                ///
                /// Marker 5
                ///
                Matrix<byte> marker5 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 5 rotated 90 degrees clockwise
                Matrix<byte> marker5Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 5 rotated 180 degrees
                Matrix<byte> marker5Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 5 rotated 270 degress clockwise
                Matrix<byte> marker5Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 5
                marker5Equal = pixelMatrix.Equals(marker5);
                // compare pixelValues with Marker2Rot90
                marker5Rot90Equal = pixelMatrix.Equals(marker5Rot90);
                // compare pixelValues with Marker2Rot180
                marker5Rot180Equal = pixelMatrix.Equals(marker5Rot180);
                // compare pixelValues with Marker2Rot270
                marker5Rot270Equal = pixelMatrix.Equals(marker5Rot270);

                ///
                /// Marker 6
                ///
                Matrix<byte> marker6 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 90 degrees clockwise
                Matrix<byte> marker6Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 180 degrees
                Matrix<byte> marker6Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 6 rotated 270 degress clockwise
                Matrix<byte> marker6Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 6
                marker6Equal = pixelMatrix.Equals(marker6);
                marker6Rot90Equal = pixelMatrix.Equals(marker6Rot90);
                marker6Rot180Equal = pixelMatrix.Equals(marker6Rot180);
                marker6Rot270Equal = pixelMatrix.Equals(marker6Rot270);

                ///
                /// Marker 7
                ///
                Matrix<byte> marker7 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 90 degrees clockwise
                Matrix<byte> marker7Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 180 degrees
                Matrix<byte> marker7Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 7 rotated 270 degress clockwise
                Matrix<byte> marker7Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 7
                marker7Equal = pixelMatrix.Equals(marker7);
                marker7Rot90Equal = pixelMatrix.Equals(marker7Rot90);
                marker7Rot180Equal = pixelMatrix.Equals(marker7Rot180);
                marker7Rot270Equal = pixelMatrix.Equals(marker7Rot270);

                ///
                /// Marker 8
                ///
                Matrix<byte> marker8 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 90 degrees clockwise
                Matrix<byte> marker8Rot90 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 180 degrees
                Matrix<byte> marker8Rot180 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // Marker 8 rotated 270 degress clockwise
                Matrix<byte> marker8Rot270 = new Matrix<byte>(new byte[,]
                {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
                });
                // compare pixelValues with Marker 8
                marker8Equal = pixelMatrix.Equals(marker8);
                marker8Rot90Equal = pixelMatrix.Equals(marker8Rot90);
                marker8Rot180Equal = pixelMatrix.Equals(marker8Rot180);
                marker8Rot270Equal = pixelMatrix.Equals(marker8Rot270);

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

                string attackValue1 = "2";
                string attackValue2 = "3";
                string attackValue3 = "5";
                string attackValue4 = "8";

                MCvScalar greenColor = new MCvScalar(0, 255, 0);
                MCvScalar blueColor = new MCvScalar(255, 0, 0);
                MCvScalar yellowColor = new MCvScalar(0, 255, 255);
                MCvScalar redColor = new MCvScalar(0, 0, 255);

                if (marker1Equal || marker1Rot90Equal || marker1Rot180Equal || marker1Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue1, greenColor);

                if (marker2Equal || marker2Rot90Equal || marker2Rot180Equal || marker2Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue2, blueColor);

                if (marker3Equal || marker3Rot90Equal || marker3Rot180Equal || marker3Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue3, yellowColor);

                if (marker4Equal || marker4Rot90Equal || marker4Rot180Equal || marker4Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue1, greenColor);

                if (marker5Equal || marker5Rot90Equal || marker5Rot180Equal || marker5Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue2, blueColor);

                if (marker6Equal || marker6Rot90Equal || marker6Rot180Equal || marker6Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue4, redColor);

                if (marker7Equal || marker7Rot90Equal || marker7Rot180Equal || marker7Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue4, redColor);

                if (marker8Equal || marker8Rot90Equal || marker8Rot180Equal || marker8Rot270Equal)
                    UtilityAR.DrawTextCube(video, intrinsic * rtMatrix, attackValue3, yellowColor);
            }

            CvInvoke.Imshow("Video", video);
        }
    }
}
