/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author McKay
 */
public class NewMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        DataSource source = new DataSource("C:\\Users\\McKay\\Documents\\iris.csv");
        Instances data = source.getDataSet();
        
        data.setClassIndex(data.numAttributes()-1);
        data.randomize(new Random(1));
        
        int trainIndex = (int) Math.round(data.numInstances() * .7);
        int testIndex = data.numInstances() - trainIndex;
        Instances trainSet = new Instances(data, 0, trainIndex);
        Instances testSet = new Instances(data, trainIndex, testIndex);
        
        HardCodedClassifier hc = new HardCodedClassifier();
        hc.buildClassifier(data);
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(hc, testSet);
        System.out.println(evaluation.toSummaryString("\n RESULTS \n", true));
        
    }
    
}
