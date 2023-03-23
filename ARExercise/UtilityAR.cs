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
        private static int frameWidth = 450;
        private static int frameHeight = 450;
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
        /// Draws a Triangle at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the Triangle onto</param>
        /// <param name="projection">The size of the Triangle</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="color">Color of the bottom of the Triangle</param>
        /// <param name="color2">Color of the pillars of the Triangle</param>
        /// <param name="color3">Color of the top of the Triangle</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawTriangle(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, MCvScalar color2, MCvScalar color3, float scale = 100)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), // bottom left point
                new Matrix<float>(new float[] { 0, scale / 2, 0, 1 }), // bottom top point
                new Matrix<float>(new float[] { scale, -scale + scale, 0, 1 }), // bottom right point
            
                new Matrix<float>(new float[] { scale, scale, scale * -1, 1 }), // bottom left pillar
                new Matrix<float>(new float[] { 0, scale / 2, scale * -1, 1 }), // top pillar
                new Matrix<float>(new float[] { scale, -scale + scale, scale * -1, 1 }), // bottom right pillar
            };

            Point[] screenPoints = worldPoints.Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), Tuple.Create(1, 2), Tuple.Create(2, 0), // bottom triangle
                Tuple.Create(0, 3), Tuple.Create(1, 4), Tuple.Create(2, 5), // triangle pillars
                Tuple.Create(3, 5), Tuple.Create(4, 3), Tuple.Create(5, 4) // top triangle
            };

            // Draw filled base
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(3).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, color, -3);

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(3).Take(3))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color2, 3);
            }

            // Draw top triangle
            foreach (Tuple<int, int> li in lineIndexes.Skip(6))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color3, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        /// <summary>
        /// Draws a Triangle2 at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the Triangle onto</param>
        /// <param name="projection">The size of the Triangle</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="color">Color of the bottom of the Triangle</param>
        /// <param name="color2">Color of the pillars of the Triangle</param>
        /// <param name="color3">Color of the top of the Triangle</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawTriangle2(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, MCvScalar color2, MCvScalar color3, float scale = 100)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), // bottom left point
                new Matrix<float>(new float[] { 0, scale / 2, 0, 1 }), // bottom top point
                new Matrix<float>(new float[] { scale, -scale + scale, 0, 1 }), // bottom right point
            
                new Matrix<float>(new float[] { scale, scale, scale * -1, 1 }), // bottom left pillar
                new Matrix<float>(new float[] { 0, scale / 2, scale * -1, 1 }), // top pillar
                new Matrix<float>(new float[] { scale, -scale + scale, scale * -1, 1 }), // bottom right pillar
            };

            Point[] screenPoints = worldPoints.Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), Tuple.Create(1, 2), Tuple.Create(2, 0), // bottom triangle
                Tuple.Create(0, 3), Tuple.Create(1, 4), Tuple.Create(2, 5), // triangle pillars
                Tuple.Create(3, 5), Tuple.Create(4, 3), Tuple.Create(5, 4) // top triangle
            };

            // Draw filled base
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(3).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, color, -3);

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(3).Take(3))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color2, 3);
            }

            // Draw top triangle
            foreach (Tuple<int, int> li in lineIndexes.Skip(6))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color3, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
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

        /// <summary>
        /// Draws a modified Cube at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the cube onto</param>
        /// <param name="projection">The size of the cube</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="color">Color of the cube's bottom</param>
        /// <param name="color2">Color2 of the cube's pillars</param>
        /// <param name="color3">Color3 of the cube's top</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawCustomCube(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, MCvScalar color2, MCvScalar color3, float scale = 100)
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
            CvInvoke.DrawContours(img, floorContour, -1, color, -3);

            // Draw top
            foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, color2, 3);
            }

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(8).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, color3, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        /// <summary>
        /// Draws a Pentagon at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the Pentagon onto</param>
        /// <param name="projection">The size of the Pentagon</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="color">Color of the bottom of the Pentagon</param>
        /// <param name="color2">Color of the pillars of the Pentagon</param>
        /// <param name="color3">Color of the top of the Pentagon</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawPentagon(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, MCvScalar color2, MCvScalar color3, float scale = 100)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, scale * 0.5f, 0, 1 }), // top point
                new Matrix<float>(new float[] { scale * 0.4f, 0, 0, 1 }), // top right point
                new Matrix<float>(new float[] { scale, scale * 0.2f, 0, 1 }), // bottom right point
                new Matrix<float>(new float[] { scale, scale * 0.8f, 0, 1 }), // bottom left point
                new Matrix<float>(new float[] { scale * 0.4f, scale, 0, 1 }), // top left point
                
                new Matrix<float>(new float[] { 0, scale * 0.5f, -scale, 1 }), // top pillar
                new Matrix<float>(new float[] { scale * 0.4f, 0 * 0.5f, -scale, 1 }), // top right pillar
                new Matrix<float>(new float[] { scale, scale * 0.2f, -scale, 1 }), // bottom right pillar
                new Matrix<float>(new float[] { scale, scale * 0.8f, -scale, 1 }), // bottom left pillar
                new Matrix<float>(new float[] { scale * 0.4f, scale, -scale, 1 }) // top left pillar
            };

            Point[] screenPoints = worldPoints.Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Bottom Pentagon
                Tuple.Create(2, 3), Tuple.Create(3, 4),
                Tuple.Create(4, 0),
                Tuple.Create(0, 5), Tuple.Create(1, 6), // Pillars
                Tuple.Create(2, 7), Tuple.Create(3, 8),
                Tuple.Create(4, 9),
                Tuple.Create(6, 5), Tuple.Create(9, 5), // Top Pentagon
                Tuple.Create(9, 8), Tuple.Create(8, 7),
                Tuple.Create(7, 6)
            };

            // Draw filled base 
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(5).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, color, -3);

            // Draw pillars 
            foreach (Tuple<int, int> li in lineIndexes.Skip(5).Take(5))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color2, 3);
            }

            // Draw top 
            foreach (Tuple<int, int> li in lineIndexes.Skip(10))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color3, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        /// <summary>
        /// Draws a Hexagon at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the Hexagon onto</param>
        /// <param name="projection">The size of the Hexagon</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="color">Color of the bottom of the hexagon</param>
        /// <param name="color2">Color of the pillars of the hexagon</param>
        /// <param name="color3">Color of the top of the hexagon</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawHexagon(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar color, MCvScalar color2, MCvScalar color3, float scale = 50)
        {

            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { scale + scale, scale, 0, 1 }), // center point
                new Matrix<float>(new float[] { scale + scale, scale, 0, 1 }), // bottom point
                new Matrix<float>(new float[] { (scale / 2) + scale, scale + 55 * MathF.Sqrt(3) / 2, 0, 1 }), // bottom left point
                new Matrix<float>(new float[] { (-scale / 2) + scale, scale + 55 * MathF.Sqrt(3) / 2, 0, 1 }), // top left point
                new Matrix<float>(new float[] { -scale + scale, scale, 0, 1 }), // top point
                new Matrix<float>(new float[] { (-scale / 2) + scale, -scale + 60 * MathF.Sqrt(3) / 2, 0, 1 }), // top right point
                new Matrix<float>(new float[] { (scale / 2) + scale, -scale + 60 * MathF.Sqrt(3) / 2, 0, 1 }), // bottom right point

                new Matrix<float>(new float[] { scale + scale, scale, scale *  -2, 1 }), // bottom pillar
                new Matrix<float>(new float[] { (scale / 2) + scale, scale + 55 * MathF.Sqrt(3) / 2, scale * -2, 1 }), // bottom left pillar
                new Matrix<float>(new float[] { (-scale / 2) + scale, scale + 55 * MathF.Sqrt(3) / 2, scale * -2, 1 }), // top left pillar
                new Matrix<float>(new float[] { -scale + scale, scale, scale * -2, 1 }), // top pillar
                new Matrix<float>(new float[] { (-scale / 2) + scale, -scale + 60 * MathF.Sqrt(3) / 2, scale * -2, 1 }), // top right pillar
                new Matrix<float>(new float[] { (scale / 2) + scale, -scale + scale, scale * -2, 1 }) // bottom right pillar
            };

            Point[] screenPoints = worldPoints.Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), // line from center
                Tuple.Create(1, 2), Tuple.Create(2, 3), // Bottom
                Tuple.Create(3, 4), Tuple.Create(4, 5), 
                Tuple.Create(5, 6), Tuple.Create(6, 1),
                Tuple.Create(7, 1), Tuple.Create(8, 2), // Pillars
                Tuple.Create(9, 3), Tuple.Create(10, 4),
                Tuple.Create(11, 5), Tuple.Create(12, 6),
                Tuple.Create(7, 8), Tuple.Create(8, 9), // Top
                Tuple.Create(9, 10), Tuple.Create(10, 11),
                Tuple.Create(11, 12), Tuple.Create(12, 7)
            };

            // Draw filled base
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(7).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, color, -3);

            //Draw Hexagon pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(7).Take(6))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color2, 3);
            }

            // Draw Hexagon top
            foreach (Tuple<int, int> li in lineIndexes.Skip(13))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color3, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        /// <summary>
        /// Draws a Pyramid at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the Pyramid onto</param>
        /// <param name="projection">The size of the Pyramid</param>
        /// <param name="attackValue">Text for total score / attack value</param>
        /// <param name="colorBase">Color of the bottom of the pyramid</param>
        /// <param name="color">Color of the pillars of the pyramid</param>
        /// <param name="scale">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        public static void DrawPyramid(IInputOutputArray img, Matrix<float> projection, string attackValue, MCvScalar colorBase, MCvScalar color, float scale = 100)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 2 }), new Matrix<float>(new float[] {scale, 0, 0, 1 }),
                new Matrix<float> (new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { scale/2, scale/2, -scale/2, 1}) // Apex of pyramid (its top)
            };

            Point[] screenPoints = worldPoints.Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Base
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(0, 4), Tuple.Create(1, 4), // Sides
                Tuple.Create(2, 4), Tuple.Create(3, 4)
            };

            // Draw filled base
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, colorBase, -3);

            // Draw sides
            foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            {
                Point point1 = screenPoints[li.Item1];
                Point point2 = screenPoints[li.Item2];

                CvInvoke.Line(img, point1, point2, color, 3);
            }

            // Add text on top of floorContour
            CvInvoke.PutText(img, attackValue, new Point((int)((screenPoints[0].X + screenPoints[2].X) / 2), (int)((screenPoints[0].Y + screenPoints[2].Y) / 2)), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 0), 3);
        }

        /// <summary>
        /// Draws a text at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the text onto</param>
        /// <param name="playerscore">Text for total score / attack value of player 1</param>
        /// /// <param name="playerscore2">Text for total score / attack value of player 2</param>
        public static void DrawText(IInputOutputArray img, string playerscore, string playerscore2)
        {
            int xOffset = 200;
            int yOffset = 160;

            // Add 2 texts on top of floorContour. One for player 1, and one for player 2.
            CvInvoke.PutText(img, playerscore, new Point((int)(frameWidth / 2) - xOffset, (int)(frameHeight / 2) - yOffset), FontFace.HersheySimplex, 2, new MCvScalar(255, 0, 0), 3);
            CvInvoke.PutText(img, playerscore2, new Point((int)(frameWidth / 2) + 320, (int)(frameHeight / 2) - yOffset), FontFace.HersheySimplex, 2, new MCvScalar(0, 0, 255), 3);
        }

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