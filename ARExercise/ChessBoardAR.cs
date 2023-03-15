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
    public class ChessBoardAR : FrameLoop
    {
        //VideoCapture.Read(IOutputArray m);
        VideoCapture vCap;
        Matrix<float> intr;
        Matrix<float> dist;
        MCvPoint3D32f[] mc;
        VectorOfPointF cornerPoints;
        PointF[] imagePoints;
        SolvePnpMethod method;

        public ChessBoardAR()
        {
            //VideoCapture vCap = new VideoCapture(0);

            //Mat frame = new Mat();

            //vCap.Read(frame);

            //CvInvoke.Imshow("bjarne", frame);

            // 1 = usb webcam, 0 = bærbar webcam
            vCap = new VideoCapture(0);
            intr = new Matrix<float>(3, 3);
            dist = new Matrix<float>(1, 5);
            UtilityAR.ReadIntrinsicsFromFile(out intr, out dist);
            mc = UtilityAR.GenerateObjectPointsForChessboard(new Size(7, 4));
            cornerPoints = new VectorOfPointF();
            imagePoints = new PointF[4];
            method = SolvePnpMethod.Iterative;
        }

        public override void OnFrame()
        {
            Mat frame = new Mat();

            vCap.Read(frame);

            //CvInvoke.FindChessboardCorners(frame, new Size(7, 4), cornerPoints, CalibCbType.NormalizeImage);

            CvInvoke.Imshow("bjarne", frame);


            bool chessFound = CvInvoke.FindChessboardCorners(frame, new Size(7, 4), cornerPoints, CalibCbType.NormalizeImage);

            if (chessFound)
            {
                Matrix<float> rotationV = new Matrix<float>(3, 1);
                Matrix<float> transV = new Matrix<float>(3, 1);
                CvInvoke.SolvePnP(mc, cornerPoints.ToArray(), intr, dist, rotationV, transV);

                CvInvoke.Rodrigues(intr, rotationV, transV);

                float[,] rValues = rotationV.Data;
                float[,] tValues = transV.Data;

                Matrix<float> rtMatrix = new Matrix<float>(new float[,]
                {
                    { rValues[0, 0], rValues[0, 1], rValues[0, 2], tValues[0, 0] },
                    { rValues[1, 0], rValues[1, 1], rValues[1, 2], tValues[1, 0] },
                    { rValues[2, 0], rValues[2, 1], rValues[2, 2], tValues[2, 0] }
                });

                //CvInvoke.Rodrigues(intr, rtMatrix);

                //CvInvoke.SolvePnP(mc, cornerPoints.ToArray(), intr, dist, rtMatrix, rtMatrix);
            }
            //CvInvoke.WaitKey();
        }
    }
}
