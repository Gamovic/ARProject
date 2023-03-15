using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ARExercise
{
    public static class UtilityAR
    {
        // Dont change these values
        private static int frameWidth = 600;
        private static int frameHeight = 600;
        // load image
        //static Mat texture = CvInvoke.Imread("four.jpg");

        /// <summary>
        /// Captures frames from the camera, whenever the user presses a button on the keyboard.<br/>
        /// Stores the images as jpg, naming them capture_0.jpg, capture_1.jpg and so on.<br/>
        /// If images already exist, they are overwritten.
        /// </summary>
        /// <param name="PatternSize">The size of the chessboard pattern to detect</param>
        /// <param name="camIndex">Index of which camera to use for capturing images</param>
        public static void CaptureLoop(Size PatternSize, int camIndex = 0)
        {
            string winName = "Preview";
            CvInvoke.NamedWindow(winName);

            using VideoCapture vcap = new VideoCapture(camIndex);

            vcap.Set(CapProp.FrameWidth, frameWidth);
            vcap.Set(CapProp.FrameHeight, frameHeight);

            int imgIndex = 0;

            while (true)
            {
                Mat frame = new Mat();
                bool frameGrabbed = vcap.Read(frame);
                if (!frameGrabbed)
                {
                    Console.WriteLine("Failed to grab frame");
                    Task.Delay(500).Wait();
                    continue;
                }

                Mat grayFrame = new Mat();
                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                Mat binaryFrame = new Mat();
                CvInvoke.Threshold(grayFrame, binaryFrame, 120, 255, ThresholdType.Otsu);

                VectorOfPointF cornerPoints = new VectorOfPointF();
                bool foundChessCorners = CvInvoke.FindChessboardCorners(binaryFrame, PatternSize, cornerPoints,
                    CalibCbType.AdaptiveThresh | CalibCbType.NormalizeImage | CalibCbType.FastCheck);

                Mat chessboardCornersFrame = new Mat();
                frame.CopyTo(chessboardCornersFrame);
                CvInvoke.DrawChessboardCorners(chessboardCornersFrame, PatternSize, cornerPoints, foundChessCorners);

                CvInvoke.SetWindowTitle(winName, $"Preview for frame {imgIndex + 1}");
                CvInvoke.Imshow(winName, chessboardCornersFrame);

                if (CvInvoke.PollKey() != -1)
                    CvInvoke.Imwrite($"capture_{imgIndex++}.jpg", frame);
            }
        }

        /// <summary>
        /// Uses all images named capture_*.jpg for calculating the calibration values.<br/>
        /// Based on <see href="https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html"/>.<br/>
        /// If successful, saves the intrinsics matrix to a file called intrinsics.json.
        /// </summary>
        /// <param name="PatternSize">The size of the chessboard pattern to detect</param>
        /// <param name="showPreview">If true, shows a preview of each image used in the calibration with a drawn overlay of the found chessboard corners</param>
        public static void CalibrateCamera(Size PatternSize, bool showPreview = true)
        {
            string[] images = Directory.GetFiles(Directory.GetCurrentDirectory(), "capture_*.jpg");

            Console.WriteLine(images.Length);

            List<MCvPoint3D32f[]> listOfObjectPoints = new List<MCvPoint3D32f[]>();
            List<VectorOfPointF> listOfCornerPoints = new List<VectorOfPointF>();

            MCvPoint3D32f[] objectPoints =  GenerateObjectPointsForChessboard(PatternSize);

            if (images.Length == 0)
            {
                Console.WriteLine("No calibration images found");
                return;
            }

            foreach (string image in images)
            {
                Mat frame = CvInvoke.Imread(image);

                Mat grayFrame = new Mat();
                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                Mat binaryFrame = new Mat();
                CvInvoke.Threshold(grayFrame, binaryFrame, 120, 255, ThresholdType.Otsu);

                VectorOfPointF cornerPoints = new VectorOfPointF();
                bool foundChessCorners = CvInvoke.FindChessboardCorners(binaryFrame, PatternSize, cornerPoints,
                    CalibCbType.AdaptiveThresh | CalibCbType.NormalizeImage | CalibCbType.FastCheck);

                if (!foundChessCorners)
                {
                    while (showPreview && CvInvoke.WaitKey(1) == -1)
                        CvInvoke.Imshow("Preview", frame);
                    continue;
                }

                CvInvoke.CornerSubPix(binaryFrame, cornerPoints, new Size(11, 11), new Size(-1, -1), new MCvTermCriteria(30, 0.1));

                listOfObjectPoints.Add(objectPoints);
                listOfCornerPoints.Add(cornerPoints);

                if (!showPreview)
                    continue;

                CvInvoke.DrawChessboardCorners(frame, PatternSize, cornerPoints, foundChessCorners);
                while (CvInvoke.WaitKey(1) == -1)
                    CvInvoke.Imshow("Preview", frame);

                
            }

            Mat intrinsics = new Mat();
            Mat distCoeffs = new Mat();

            Size frameSize = new Size(frameWidth, frameHeight);

            double reprojectionError = CvInvoke.CalibrateCamera(listOfObjectPoints.ToArray(), listOfCornerPoints.Select(x => x.ToArray()).ToArray(),
                frameSize,
                intrinsics,
                distCoeffs,
                CalibType.Default,
                new MCvTermCriteria(30, 0.1),
                out Mat[] rVecs, out Mat[] tVecs);

            using FileStorage fs = new FileStorage("intrinsics.json", FileStorage.Mode.Write);
            fs.Write(intrinsics, "Intrinsics");
            fs.Write(distCoeffs, "DistCoeffs");
        }

        /// <summary>
        /// Reads and returns the intrinsics matrices from the file intrinsics.json.<br/>
        /// See <see cref="CalibrateCamera"/>.
        /// </summary>
        /// <param name="intrinsics">The resulting intrinsics read from the file</param>
        /// <param name="distCoeffs">The resulting distortion coefficients read from the file</param>
        public static void ReadIntrinsicsFromFile(out Matrix<float> intrinsics, out Matrix<float> distCoeffs)
        {
            Mat intrinsicsMat = new Mat();
            Mat distCoeffsMat = new Mat();

            using FileStorage fs = new FileStorage("intrinsics.json", FileStorage.Mode.Read);

            FileNode intrinsicsNode = fs.GetNode("Intrinsics");
            FileNode distCoeffsNode = fs.GetNode("DistCoeffs");

            intrinsicsNode.ReadMat(intrinsicsMat);
            distCoeffsNode.ReadMat(distCoeffsMat);

            intrinsics = new Matrix<float>(3, 3);
            distCoeffs = new Matrix<float>(1, 5);

            intrinsicsMat.ConvertTo(intrinsics, DepthType.Cv32F);
            distCoeffsMat.ConvertTo(distCoeffs, DepthType.Cv32F);
        }

        public static MCvPoint3D32f[] GenerateObjectPointsForChessboard(Size PatternSize)
        {
            List<MCvPoint3D32f> objPoints = new List<MCvPoint3D32f>();
            for (int y = 0; y < PatternSize.Height; y++)
            {
                for (int x = 0; x < PatternSize.Width; x++)
                {
                    objPoints.Add(new MCvPoint3D32f(x, y, 0));
                }
            }

            return objPoints.ToArray();
        }

        /// <summary>
        /// Draws a cube at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the cube onto</param>
        /// <param name="scale">The size of the cube</param>
        /// <param name="projection">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawCube(IInputOutputArray img, Matrix<float> projection, float scale = 100)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            // Draw filled floor
            VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            CvInvoke.DrawContours(img, floorContour, -1, new MCvScalar(0, 255, 0), -3);
            //CvInvoke.DrawContours(img, floorContour, -1, new Bgr(0, 255, 0).MCvScalar, -1);

            // Draw top
            foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, new MCvScalar(255, 0, 0), 3);
            }

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(8).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, new MCvScalar(0, 0, 255), 3);
            }
        }

        public static void DrawText(IInputOutputArray img, Matrix<float> projection, string text, float scale = 1)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            // Add text on top of floorContour
            CvInvoke.PutText(img, text, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(255, 0, 0), 3);

        }





        public static void DrawNewCube(IInputOutputArray img, Matrix<float> projection, float scale = 1)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            // Draw filled floor
            //VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            //Image<Bgr, byte> floorImg = new Image<Bgr, byte>("four.jpg");
            //Rectangle floorRect = CvInvoke.BoundingRectangle(floorContour);
            //floorImg = floorImg.Resize(floorRect.Width, floorRect.Height, Inter.Linear);
            //Mat imgROI = new Mat(img, floorRect);
            //floorImg.CopyTo(img);
            //img.ROI = Rectangle.Empty;

            VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            CvInvoke.DrawContours(img, floorContour, -1, new MCvScalar(255, 255, 255), -3);
            //CvInvoke.DrawContours(img, floorContour, -1, new Bgr(0, 255, 0).MCvScalar, -1);

            // Calculate angle of bottom edge
            //double angle = Math.Atan2(screenPoints[3].Y - screenPoints[0].Y, screenPoints[3].X - screenPoints[0].X) * 180 / Math.PI;
            //int i;
            //// Add text on top of floorContour with same angle
            //Size textSize = CvInvoke.GetTextSize("1", FontFace.HersheySimplex, 2, 3, ref);
            //Point textLocation = new Point((int)((screenPoints[0].X + screenPoints[2].X - textSize.Width) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y + baseline) / 2));
            //Mat rotMat = CvInvoke.GetRotationMatrix2D(textLocation, angle, 1);
            //CvInvoke.PutText(img, "1", textLocation, FontFace.HersheySimplex, 2, new MCvScalar(255, 0, 0), 3);
            //CvInvoke.WarpAffine(img, img, rotMat, img.Size);

            //// Add text on top of floorContour
            //CvInvoke.PutText(img, "4", new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(255, 0, 0), 3);


            // Draw top
            //foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            //{
            //    Point p1 = screenPoints[li.Item1];
            //    Point p2 = screenPoints[li.Item2];

            //    CvInvoke.Line(img, p1, p2, new MCvScalar(255, 0, 0), 3);
            //}

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(8).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, new MCvScalar(0, 0, 255), 3);
            }
        }







        public static void DrawRedCube(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, float scale = 1)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            //var imageFilePath = ("four2.jpg");

            //Mat image = CvInvoke.Imread(imageFilePath, ImreadModes.Color);

            //Mat imageMat = CvInvoke.Imread("four.jpg");
            //IInputOutputArray imageArray = (IInputOutputArray)imageMat.GetInputOutputArray();

            //Size inputSize = new Size(image.Width, image.Height);
            //CvInvoke.Resize(image, image, inputSize);

            //double alpha = 0.5f;
            //CvInvoke.AddWeighted(img, alpha, img, 1 - alpha, 0, img);

            //Rectangle floorRect = new Rectangle((int)screenPoints[0].X, (int)screenPoints[0].Y, (int)(screenPoints[2].X - screenPoints[0].X), (int)(screenPoints[2].Y - screenPoints[0].Y));
            //CvInvoke.Rectangle(texture, floorRect, new MCvScalar(0, 255, 0), -1);
            // Draw filled floor
            VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            CvInvoke.DrawContours(img, floorContour, -1, color, -3);

            //CvInvoke.DrawMarker(texture, screenPoints[0], new MCvScalar(0,0,255), MarkerTypes.Square);

            // Draw top
            foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, color, 3);
            }

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(8).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, color, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        //public static void DrawTextureCube(IInputOutputArray img, Matrix<float> projection, float scale = 1)
        //{
            //Matrix<float>[] worldPoints = new[]
            //{
            //    new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
            //    new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
            //    new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
            //    new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            //};

            //Point[] screenPoints = worldPoints
            //    .Select(x => WorldToScreen(x, projection)).ToArray();

            //Tuple<int, int>[] lineIndexes = new[] {
            //    Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
            //    Tuple.Create(2, 3), Tuple.Create(3, 0),
            //    Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
            //    Tuple.Create(6, 7), Tuple.Create(7, 4),
            //    Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
            //    Tuple.Create(2, 6), Tuple.Create(3, 7)
            //};

            //// Draw filled floor
            //VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            //CvInvoke.DrawContours(img, floorContour, -1, new MCvScalar(0, 255, 0), -3);

            //// Draw top and sides with textures
            //Size imageSize = img.GetInputArray().GetSize();
            //Size faceSize = new Size((int)Math.Round(Math.Sqrt(imageSize.Width * imageSize.Height / 6.0)), (int)Math.Round(Math.Sqrt(imageSize.Width * imageSize.Height / 6.0)));
            //for (int i = 0; i < 6; i++)
            //{
            //    if (i == 1 || i == 4)
            //    {
            //        // Skip sides
            //        continue;
            //    }

            //    PointF[] points = new PointF[]
            //    {
            //        screenPoints[lineIndexes[i * 2].Item1], screenPoints[lineIndexes[i * 2].Item2],
            //        screenPoints[lineIndexes[i * 2 + 1].Item2], screenPoints[lineIndexes[i * 2 + 1].Item1]
            //    };
            //    Mat face = new Mat(faceSize, texture.Depth, texture.NumberOfChannels);
            //    CvInvoke.Resize(texture, face, faceSize, 0, 0, Inter.Linear);

            //    Point[] texCoords = new Point[] { new Point(0, 0), new Point(faceSize.Width, 0), new Point(faceSize.Width, faceSize.Height), new Point(0, faceSize.Height) };
            //    VectorOfPoint texCoordsVector = new VectorOfPoint(texCoords);
            //    CvInvoke.FillConvexPoly(face, texCoordsVector, new MCvScalar(255, 255, 255));
            //    CvInvoke.FillConvexPoly(face, texCoordsVector, new MCvScalar(255, 255, 255));
            //    PointF[] srcPts = new PointF[] { new PointF(0, 0), new PointF(faceSize.Width, 0), new PointF(faceSize.Width, faceSize.Height), new PointF(0, faceSize.Height) };
            //    Mat homography = CvInvoke.FindHomography(points, srcPts, RobustEstimationAlgorithm.Ransac);
            //    CvInvoke.WarpPerspective(face, img, homography, imageSize, Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar());
            //    Rectangle textureRect = new Rectangle((int)points[0].X, (int)points[0].Y, (int)(points[2].X - points[0].X), (int)(points[2].Y - points[0].Y));
            //    CvInvoke.Rectangle(img, textureRect, new MCvScalar(0, 0, 0), 3);
            //}
        //}

        /// <summary>
        /// Converts a homogeneous world coordinate to a screen point
        /// </summary>
        /// <param name="worldPoint">The homogeneous world coordinate</param>
        /// <param name="projection">The projection-matrix to use for converting world coordinates to screen coordinates</param>
        /// <returns>The Point in screen coordinates</returns>
        public static Point WorldToScreen(Matrix<float> worldPoint, Matrix<float> projection)
        {
            Matrix<float> result = projection * worldPoint;
            return new Point((int)(result[0, 0] / result[2, 0]), (int)(result[1, 0] / result[2, 0]));
        }
    }
}