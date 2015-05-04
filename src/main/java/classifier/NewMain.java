/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author McKay
 */
public class NewMain {

    private final static String filepath = "C:\\Users\\McKay\\Documents\\";
    private static String filename;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        for (int i = 0; i < 3; ++i)
        {
            if (i==0)
                filename = "iris.csv";
            else if (i==1)
                filename = "car.csv";
            else
                filename = "cancer.csv";
            
            System.out.println("============= "+filename+" =============\n");
                        
            DataSource source = new DataSource(filepath+filename);
            Instances dataset = source.getDataSet();
            Standardize standard = new Standardize();
            standard.setInputFormat(dataset);
            Instances data = Filter.useFilter(dataset, standard);

            data.setClassIndex(data.numAttributes()-1);
            data.randomize(new Random(1));

            int trainIndex = (int) Math.round(data.numInstances() * .8);
            int testIndex = data.numInstances() - trainIndex;
            Instances trainSet = new Instances(data, 0, trainIndex);
            Instances testSet = new Instances(data, trainIndex, testIndex);

            for (int k = 1; k < 6; ++k)
            {
                System.out.println("\n\n\t k = " +k);
                Classifier hc = new KNNClassifier(k);
                hc.buildClassifier(data);
                Evaluation evaluation = new Evaluation(trainSet);
                evaluation.evaluateModel(hc, testSet);
                System.out.println(evaluation.toSummaryString("\n RESULTS \n", true));
            }
            
            System.out.println("\n\n");
            
        }
        
    }
    
}
