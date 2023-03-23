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
        Matrix<float> rotationMatrix = new Matrix<float>(3, 3);

        Matrix<byte> pixelMatrix;

        Matrix<float> rtMatrix;

        VectorOfVectorOfPoint contours;

        VectorOfVectorOfPoint squareContours;

        #region Marker bools
        bool marker1Equal, marker1Rot90Equal, marker1Rot180Equal, marker1Rot270Equal;
        bool marker2Equal, marker2Rot90Equal, marker2Rot180Equal, marker2Rot270Equal;
        bool marker3Equal, marker3Rot90Equal, marker3Rot180Equal, marker3Rot270Equal;
        bool marker4Equal, marker4Rot90Equal, marker4Rot180Equal, marker4Rot270Equal;
        bool marker5Equal, marker5Rot90Equal, marker5Rot180Equal, marker5Rot270Equal;
        bool marker6Equal, marker6Rot90Equal, marker6Rot180Equal, marker6Rot270Equal;
        bool marker7Equal, marker7Rot90Equal, marker7Rot180Equal, marker7Rot270Equal;
        bool marker8Equal, marker8Rot90Equal, marker8Rot180Equal, marker8Rot270Equal;
        #endregion

        int attackValue1, attackValue2, attackValue3, attackValue4,
            attackValue5, attackValue6, attackValue7, attackValue8;

        int totalAttackValue;
        List<int> attackValues;

        //MCvScalar colour;
        MCvScalar greenColor, blueColor, yellowColor, redColor;

        #region Marker definitions
        Matrix<byte> marker1;
        Matrix<byte> marker1Rot90;
        Matrix<byte> marker1Rot180;
        Matrix<byte> marker1Rot270;

        Matrix<byte> marker2;
        Matrix<byte> marker2Rot90;
        Matrix<byte> marker2Rot180;
        Matrix<byte> marker2Rot270;

        Matrix<byte> marker3;
        Matrix<byte> marker3Rot90;
        Matrix<byte> marker3Rot180;
        Matrix<byte> marker3Rot270;

        Matrix<byte> marker4;
        Matrix<byte> marker4Rot90;
        Matrix<byte> marker4Rot180;
        Matrix<byte> marker4Rot270;

        Matrix<byte> marker5;
        Matrix<byte> marker5Rot90;
        Matrix<byte> marker5Rot180;
        Matrix<byte> marker5Rot270;

        Matrix<byte> marker6;
        Matrix<byte> marker6Rot90;
        Matrix<byte> marker6Rot180;
        Matrix<byte> marker6Rot270;

        Matrix<byte> marker7;
        Matrix<byte> marker7Rot90;
        Matrix<byte> marker7Rot180;
        Matrix<byte> marker7Rot270;

        Matrix<byte> marker8;
        Matrix<byte> marker8Rot90;
        Matrix<byte> marker8Rot180;
        Matrix<byte> marker8Rot270;
        #endregion

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

            attackValue1 = 2;
            attackValue2 = 3;
            attackValue3 = 5;
            attackValue4 = 2;
            attackValue5 = 3;
            attackValue6 = 8;
            attackValue7 = 8;
            attackValue8 = 5;

            totalAttackValue = 0;
            // List to store the attack values in
            attackValues = new List<int>();

            greenColor = new MCvScalar(0, 255, 0);
            blueColor = new MCvScalar(255, 0, 0);
            yellowColor = new MCvScalar(0, 255, 255);
            redColor = new MCvScalar(0, 0, 255);

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
                /// Marker matrices for detection
                ///
                #region Marker matrices
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
                #endregion

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

                // Convert rotation vector to rotation matrix
                CvInvoke.Rodrigues(rotationVector, rotationMatrix);

                // New matrix from our new rotaion matrix's data and translation data
                float[,] rValues = rotationMatrix.Data;
                float[,] tValues = translationVector.Data;

                rtMatrix = new Matrix<float>(new float[,]
                {
                        { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
                        { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
                        { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
                });

                ///
                /// Draw cubes, pyramides, hexagons
                ///
                #region Draw geometrical shapes
                ///
                /// Check marker 1 and draw cube if pixelMatrix equals marker 1
                ///
                if (marker1Equal)
                {
                    Console.WriteLine("Marker1 and pMatrix are equal");
                    UtilityAR.DrawTriangle(image, intrinsic * rtMatrix, attackValue1.ToString(), blueColor, yellowColor, redColor);
                }
                if (marker1Rot90Equal)
                {
                    Console.WriteLine("Marker1Rot90 and pMatrix are equal");
                    UtilityAR.DrawTriangle(image, intrinsic * rtMatrix, attackValue1.ToString(), blueColor, yellowColor, redColor);
                }
                if (marker1Rot180Equal)
                {
                    Console.WriteLine("Marker1Rot180 and pMatrix are equal");
                    UtilityAR.DrawTriangle(image, intrinsic * rtMatrix, attackValue1.ToString(), blueColor, yellowColor, redColor);
                }
                if (marker1Rot270Equal)
                {
                    Console.WriteLine("Marker1Rot270 and pMatrix are equal");
                    UtilityAR.DrawTriangle(image, intrinsic * rtMatrix, attackValue1.ToString(), blueColor, yellowColor, redColor);
                }

                ///
                /// Check marker 2 and draw cube if pixelMatrix equals marker 2
                ///
                if (marker2Equal)
                {
                    Console.WriteLine("Marker2 and pMatrix are equal");
                    UtilityAR.DrawHexagon(image, intrinsic * rtMatrix, attackValue4.ToString(), redColor, blueColor, yellowColor);
                }
                if (marker2Rot90Equal)
                {
                    Console.WriteLine("Marker2Rot90 and pMatrix are equal");
                    UtilityAR.DrawHexagon(image, intrinsic * rtMatrix, attackValue4.ToString(), redColor, blueColor, yellowColor);
                }
                if (marker2Rot180Equal)
                {
                    Console.WriteLine("Marker2Rot180 and pMatrix are equal");
                    UtilityAR.DrawHexagon(image, intrinsic * rtMatrix, attackValue4.ToString(), redColor, blueColor, yellowColor);
                }
                if (marker2Rot270Equal)
                {
                    Console.WriteLine("Marker2Rot270 and pMatrix are equal");
                    UtilityAR.DrawHexagon(image, intrinsic * rtMatrix, attackValue4.ToString(), redColor, blueColor, yellowColor);
                }

                ///
                /// Check marker 3 and draw cube if pixelMatrix equals marker 3
                ///
                if (marker3Equal)
                {
                    Console.WriteLine("Marker3 and pMatrix are equal");
                    UtilityAR.DrawPentagon(image, intrinsic * rtMatrix, attackValue3.ToString(), blueColor, yellowColor, greenColor);
                }
                if (marker3Rot90Equal)
                {
                    Console.WriteLine("Marker3Rot90 and pMatrix are equal");
                    UtilityAR.DrawPentagon(image, intrinsic * rtMatrix, attackValue3.ToString(), blueColor, yellowColor, greenColor);
                }
                if (marker3Rot180Equal)
                {
                    Console.WriteLine("Marker3Rot180 and pMatrix are equal");
                    UtilityAR.DrawPentagon(image, intrinsic * rtMatrix, attackValue3.ToString(), blueColor, yellowColor, greenColor);
                }
                if (marker3Rot270Equal)
                {
                    Console.WriteLine("Marker3Rot270 and pMatrix are equal");
                    UtilityAR.DrawPentagon(image, intrinsic * rtMatrix, attackValue3.ToString(), blueColor, yellowColor, greenColor);
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
                    UtilityAR.DrawPyramid(image, intrinsic * rtMatrix, attackValue3.ToString(), greenColor, redColor);
                }
                if (marker7Rot90Equal)
                {
                    Console.WriteLine("Marker7Rot90 and pMatrix are equal");
                    UtilityAR.DrawPyramid(image, intrinsic * rtMatrix, attackValue3.ToString(), greenColor, redColor);
                }
                if (marker7Rot180Equal)
                {
                    Console.WriteLine("Marker7Rot180 and pMatrix are equal");
                    UtilityAR.DrawPyramid(image, intrinsic * rtMatrix, attackValue3.ToString(), greenColor, redColor);
                }
                if (marker7Rot270Equal)
                {
                    Console.WriteLine("Marker7Rot270 and pMatrix are equal");
                    UtilityAR.DrawPyramid(image, intrinsic * rtMatrix, attackValue3.ToString(), greenColor, redColor);
                }

                ///
                /// Check marker 8 and draw cube if pixelMatrix equals marker 8
                ///
                if (marker8Equal)
                {
                    Console.WriteLine("Marker8 and pMatrix are equal");
                    UtilityAR.DrawCustomCube(image, intrinsic * rtMatrix, attackValue2.ToString(), yellowColor, blueColor, redColor);
                }
                if (marker8Rot90Equal)
                {
                    Console.WriteLine("Marker8Rot90 and pMatrix are equal");
                    UtilityAR.DrawCustomCube(image, intrinsic * rtMatrix, attackValue2.ToString(), yellowColor, blueColor, redColor);
                }
                if (marker8Rot180Equal)
                {
                    Console.WriteLine("Marker8Rot180 and pMatrix are equal");
                    UtilityAR.DrawCustomCube(image, intrinsic * rtMatrix, attackValue2.ToString(), yellowColor, blueColor, redColor);
                }
                if (marker8Rot270Equal)
                {
                    Console.WriteLine("Marker8Rot270 and pMatrix are equal");
                    UtilityAR.DrawCustomCube(image, intrinsic * rtMatrix, attackValue2.ToString(), yellowColor, blueColor, redColor);
                }
                #endregion
            }
            CvInvoke.Imshow("draw cube", image);

            GetMarkers();
        }

        /// <summary>
        /// Sets the definitions for the markers, to be used in the update loop
        /// </summary>
        public void GetMarkers()
        {
            ///
            /// Marker 1
            /// 
            marker1 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 1 rotated 90 degrees clockwise
            marker1Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 1 rotated 180 degrees
            marker1Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 1 rotated 270 degress clockwise
            marker1Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 2
            /// 
            marker2 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 2 rotated 90 degrees clockwise
            marker2Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 2 rotated 180 degrees
            marker2Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 2 rotated 270 degress clockwise
            marker2Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 3 normal
            /// 
            marker3 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 3 rotated 90 degrees clockwise
            marker3Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 3 rotated 180 degrees
            marker3Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 3 rotated 270 degress clockwise
            marker3Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 4
            ///
            marker4 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 4 rotated 90 degrees clockwise
            marker4Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 4 rotated 180 degrees
            marker4Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 4 rotated 270 degress clockwise
            marker4Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 5
            ///
            marker5 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 5 rotated 90 degrees clockwise
            marker5Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 5 rotated 180 degrees
            marker5Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 5 rotated 270 degress clockwise
            marker5Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 6
            ///
            marker6 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 6 rotated 90 degrees clockwise
            marker6Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 6 rotated 180 degrees
            marker6Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 6 rotated 270 degress clockwise
            marker6Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 7
            ///
            marker7 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 7 rotated 90 degrees clockwise
            marker7Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 7 rotated 180 degrees
            marker7Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 7 rotated 270 degress clockwise
            marker7Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });

            ///
            /// Marker 8
            ///
            marker8 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 8 rotated 90 degrees clockwise
            marker8Rot90 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255,   0, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 8 rotated 180 degrees
            marker8Rot180 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255,   0, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
            // Marker 8 rotated 270 degress clockwise
            marker8Rot270 = new Matrix<byte>(new byte[,]
            {
                    { 0,   0,   0,   0,   0, 0 },
                    { 0, 255, 255,   0, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0, 255, 255, 255, 255, 0 },
                    { 0,   0,   0,   0,   0, 0 }
            });
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
                #region Markers for detection
                ///
                /// Marker 1
                /// 

                // compare pixelValues with Marker2
                marker1Equal = pixelMatrix.Equals(marker1);
                marker1Rot90Equal = pixelMatrix.Equals(marker1Rot90);
                marker1Rot180Equal = pixelMatrix.Equals(marker1Rot180);
                marker1Rot270Equal = pixelMatrix.Equals(marker1Rot270);

                ///
                /// Marker 2
                /// 

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

                // compare pixelValues with Marker3
                marker3Equal = pixelMatrix.Equals(marker3);
                // compare pixelValues with Marker3Rot90
                marker3Rot90Equal = pixelMatrix.Equals(marker3Rot90);
                // compare pixelValues with Marker3Rot180
                marker3Rot180Equal = pixelMatrix.Equals(marker3Rot180);
                // compare pixelValues with Marker3Rot270
                marker3Rot270Equal = pixelMatrix.Equals(marker3Rot270);

                ///
                /// Marker 4
                ///

                // compare pixelValues with Marker3
                marker4Equal = pixelMatrix.Equals(marker4);
                // compare pixelValues with Marker3Rot90
                marker4Rot90Equal = pixelMatrix.Equals(marker4Rot90);
                // compare pixelValues with Marker3Rot180
                marker4Rot180Equal = pixelMatrix.Equals(marker4Rot180);
                // compare pixelValues with Marker3Rot270
                marker4Rot270Equal = pixelMatrix.Equals(marker4Rot270);

                ///
                /// Marker 5
                ///

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

                // compare pixelValues with Marker 6
                marker6Equal = pixelMatrix.Equals(marker6);
                marker6Rot90Equal = pixelMatrix.Equals(marker6Rot90);
                marker6Rot180Equal = pixelMatrix.Equals(marker6Rot180);
                marker6Rot270Equal = pixelMatrix.Equals(marker6Rot270);

                ///
                /// Marker 7
                ///

                // compare pixelValues with Marker 7
                marker7Equal = pixelMatrix.Equals(marker7);
                marker7Rot90Equal = pixelMatrix.Equals(marker7Rot90);
                marker7Rot180Equal = pixelMatrix.Equals(marker7Rot180);
                marker7Rot270Equal = pixelMatrix.Equals(marker7Rot270);

                ///
                /// Marker 8
                ///

                // compare pixelValues with Marker 8
                marker8Equal = pixelMatrix.Equals(marker8);
                marker8Rot90Equal = pixelMatrix.Equals(marker8Rot90);
                marker8Rot180Equal = pixelMatrix.Equals(marker8Rot180);
                marker8Rot270Equal = pixelMatrix.Equals(marker8Rot270);

                #endregion

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


                CvInvoke.Rodrigues(rotationVector, rotationMatrix);

                // New matrix from our new rotaion matrix's data and translation data
                float[,] rValues = rotationMatrix.Data;
                float[,] tValues = translationVector.Data;

                Matrix<float> rtMatrix = new Matrix<float>(new float[,]
                {
                        { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
                        { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
                        { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
                });




                ///
                /// Draw
                ///
                #region Draw geometrical Shapes
                if (marker1Equal || marker1Rot90Equal || marker1Rot180Equal || marker1Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue1.ToString());
                    //totalAttackValue += attackValue1;
                    if (!attackValues.Contains(attackValue1))
                    {

                        UtilityAR.DrawTriangle(video, intrinsic * rtMatrix, attackValue1.ToString(), greenColor, redColor, blueColor);
                        attackValues.Add(attackValue1);
                    }
                }

                if (marker2Equal || marker2Rot90Equal || marker2Rot180Equal || marker2Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue2.ToString());
                    //totalAttackValue += attackValue2;
                    if (!attackValues.Contains(attackValue2))
                    {

                        UtilityAR.DrawCustomCube(video, intrinsic * rtMatrix, attackValue2.ToString(), blueColor, yellowColor, greenColor);
                        attackValues.Add(attackValue2);
                    }
                }

                if (marker3Equal || marker3Rot90Equal || marker3Rot180Equal || marker3Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue3.ToString());
                    //totalAttackValue += attackValue3;
                    if (!attackValues.Contains(attackValue3))
                    {

                        UtilityAR.DrawPentagon(video, intrinsic * rtMatrix, attackValue3.ToString(), yellowColor, redColor, blueColor);
                        attackValues.Add(attackValue3);
                    }
                }

                if (marker4Equal || marker4Rot90Equal || marker4Rot180Equal || marker4Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue1.ToString());
                    //totalAttackValue += attackValue1;
                    if (!attackValues.Contains(attackValue4))
                    {

                        UtilityAR.DrawTriangle(video, intrinsic * rtMatrix, attackValue4.ToString(), greenColor, redColor, blueColor);
                        attackValues.Add(attackValue4);
                    }
                }

                if (marker5Equal || marker5Rot90Equal || marker5Rot180Equal || marker5Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue2.ToString());
                    //totalAttackValue += attackValue2;
                    if (!attackValues.Contains(attackValue5))
                    {

                        UtilityAR.DrawCustomCube(video, intrinsic * rtMatrix, attackValue5.ToString(), blueColor, yellowColor, greenColor);
                        attackValues.Add(attackValue5);
                    }
                }

                if (marker6Equal || marker6Rot90Equal || marker6Rot180Equal || marker6Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue4.ToString());
                    //totalAttackValue += attackValue4;
                    if (!attackValues.Contains(attackValue6))
                    {

                        UtilityAR.DrawHexagon(video, intrinsic * rtMatrix, attackValue6.ToString(), redColor, greenColor, yellowColor);
                        attackValues.Add(attackValue6);
                    }
                }

                if (marker7Equal || marker7Rot90Equal || marker7Rot180Equal || marker7Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue4.ToString());
                    //totalAttackValue += attackValue4;
                    if (!attackValues.Contains(attackValue7))
                    {

                        UtilityAR.DrawHexagon(video, intrinsic * rtMatrix, attackValue7.ToString(), redColor, greenColor, yellowColor);
                        attackValues.Add(attackValue7);
                    }
                }

                if (marker8Equal || marker8Rot90Equal || marker8Rot180Equal || marker8Rot270Equal)
                {
                    //totalAttackValue += int.Parse(attackValue3.ToString());
                    //totalAttackValue += attackValue3;
                    if (!attackValues.Contains(attackValue8))
                    {
                        UtilityAR.DrawPentagon(video, intrinsic * rtMatrix, attackValue3.ToString(), yellowColor, redColor, blueColor);
                        attackValues.Add(attackValue8);

                    }
                }
                #endregion


            }
            totalAttackValue = attackValues.Sum();



            // calculate the total attack value by summing up all the attack values in the list
            //int totalAttackVal = attackValues.Sum();

            //UtilityAR.DrawCustomCube(video, intrinsic * rtMatrix, attackValue, colour);
            UtilityAR.DrawText(video, totalAttackValue.ToString(), new string("0"));
            //Console.WriteLine(totalAttackValue);

            CvInvoke.Imshow("Video", video);
        }
    }
}
