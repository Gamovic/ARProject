using System.Drawing;

namespace ARExercise
{
    internal class Program
    {

        static void Main(string[] args)
        {
            //UtilityAR.CaptureLoop(new Size(7, 4), 1);
            //UtilityAR.CalibrateCamera(new Size(7, 4), true);

            MarkerAR newMarkerAR = new MarkerAR();
            newMarkerAR.Run();
        }
    }
}