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
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author McKay
 */
public class NewMain {

    private final static String filepath = "C:\\Users\\McKay\\Documents\\";
    private static String filename = "lenses.csv";
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
       
            
        System.out.println("============= "+filename+" =============\n");

        DataSource source = new DataSource(filepath+filename);
        Instances dataSetPre = source.getDataSet();

        dataSetPre.setClassIndex(dataSetPre.numAttributes() - 1);

        Standardize stand = new Standardize();
        stand.setInputFormat(dataSetPre);

        Discretize discretize = new Discretize();
        discretize.setInputFormat(dataSetPre);

        Instances dataSet = dataSetPre;

        dataSet = Filter.useFilter(dataSet, discretize);
        dataSet = Filter.useFilter(dataSet, stand);

        dataSet.randomize(new Random(9001));

        Classifier classifier = new ID3();
        Evaluation eval = new Evaluation(dataSet);
        final int folds = 10;
        
        for (int n = 0; n < folds; n++) {
            Instances train = dataSet.trainCV(folds, n);
            Instances test = dataSet.testCV(folds, n);

            Classifier clsCopy = Classifier.makeCopy(classifier);
            clsCopy.buildClassifier(train);
            eval.evaluateModel(clsCopy, test);
        }
        
    }
    
}
