using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.IO;
using System.Linq;
using System.Collections.Generic;
namespace MovieRecomendationApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var TrainingDataLocation = @"C:\Users\gravi\Downloads\recommendation-ratings-train.csv";
            var TestDataLocation = @"C:\Users\gravi\Downloads\recommendation-ratings-test.csv";
            //STEP 1: Create MLContext to be shared across the model creation workflow objects 

            MLContext mlcontext = new MLContext();



            //STEP 2: Read the training data which will be used to train the movie recommendation model    

            //The schema for training data is defined by type 'TInput' in LoadFromTextFile<TInput>() method.

            IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TrainingDataLocation, hasHeader: true, separatorChar: ',');



            //STEP 3: Transform your data by encoding the two features userId and movieID. These encoded features will be provided as input

            //        to our MatrixFactorizationTrainer.

            var dataProcessingPipeline = mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: nameof(MovieRating.userId))

                           .Append(mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: nameof(MovieRating.movieId)));



            //Specify the options for MatrixFactorization trainer            

            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();

            options.MatrixColumnIndexColumnName = "userIdEncoded";

            options.MatrixRowIndexColumnName = "movieIdEncoded";

            options.LabelColumnName = "Label";

            options.NumberOfIterations = 20;

            options.ApproximationRank = 100;



            //STEP 4: Create the training pipeline 

            var trainingPipeLine = dataProcessingPipeline.Append(mlcontext.Recommendation().Trainers.MatrixFactorization(options));



            //STEP 5: Train the model fitting to the DataSet

            Console.WriteLine("=============== Training the model ===============");

            ITransformer model = trainingPipeLine.Fit(trainingDataView);



            //STEP 6: Evaluate the model performance 

            Console.WriteLine("=============== Evaluating the model ===============");

            IDataView testDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true, separatorChar: ',');

            var prediction = model.Transform(testDataView);

            var metrics = mlcontext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);



            //STEP 7:  Try/test a single prediction by predicting a single movie rating for a specific user

            var predictionengine = mlcontext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            /* Make a single movie rating prediction, the scores are for a particular user and will range from 1 - 5. 

               The higher the score the higher the likelyhood of a user liking a particular movie.

               You can recommend a movie to a user if say rating > 3.5.*/
            var predictionuserId = 6;
            var predictionmovieId = 10;
            var movieratingprediction = predictionengine.Predict(

                new MovieRating()

                {

                    //Example rating prediction for userId = 6, movieId = 10 (GoldenEye)

                    userId = predictionuserId,

                    movieId = predictionmovieId

                }

            );



            Movie movieService = new Movie();

            Console.WriteLine("For userId:" + predictionuserId + " movie rating prediction (1 - 5 stars) for movie:" + movieService.Get(predictionmovieId).movieTitle + " is:" + Math.Round(movieratingprediction.Score, 1));



            Console.WriteLine("=============== End of process, hit any key to finish ===============");

            Console.ReadLine();

        }

        public static string GetAbsolutePath(string relativePath)

        {

            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;



            string fullPath = Path.Combine(assemblyFolderPath, relativePath);



            return fullPath;

        }
    }
    public class Movie

    {

        public int movieId;



        public String movieTitle;



        private static String moviesdatasetRelativepath = @"C:\Users\gravi\Downloads\recommendation-movies.csv";

        private static string moviesdatasetpath = Program.GetAbsolutePath(moviesdatasetRelativepath);



        public Lazy<List<Movie>> _movies = new Lazy<List<Movie>>(() => LoadMovieData(moviesdatasetpath));



        public Movie()

        {

        }



        public Movie Get(int id)

        {

            return _movies.Value.Single(m => m.movieId == id);

        }



        private static List<Movie> LoadMovieData(String moviesdatasetpath)

        {

            var result = new List<Movie>();

            Stream fileReader = File.OpenRead(moviesdatasetpath);

            StreamReader reader = new StreamReader(fileReader);

            try

            {

                bool header = true;

                int index = 0;

                var line = "";

                while (!reader.EndOfStream)

                {

                    if (header)

                    {

                        line = reader.ReadLine();

                        header = false;

                    }

                    line = reader.ReadLine();

                    string[] fields = line.Split(',');

                    int movieId = Int32.Parse(fields[0].ToString().TrimStart(new char[] { '0' }));

                    string movieTitle = fields[1].ToString();

                    result.Add(new Movie() { movieId = movieId, movieTitle = movieTitle });

                    index++;

                }

            }

            finally

            {

                if (reader != null)

                {

                    reader.Dispose();

                }

            }



            return result;

        }

    }
    public class MovieRatingPrediction

    {

        public float Label;



        public float Score;

    }
    public class MovieRating

    {

        [LoadColumn(0)]

        public float userId;



        [LoadColumn(1)]

        public float movieId;



        [LoadColumn(2)]

        public float Label;

    }
}
